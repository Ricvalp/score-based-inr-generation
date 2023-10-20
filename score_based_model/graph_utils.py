# Typing
from collections import namedtuple
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp

# Jraph
import jraph
import numpy as np
from ml_collections import ConfigDict
from ml_collections import FrozenConfigDict as FrozenDict


#############################################
# Nef utils
#############################################
def flatten_dict(d: Dict, separation: str = "."):
    """Flattens a dictionary.

    Args:
        d (Dict): The dictionary to flatten.

    Returns:
        Dict: The flattened dictionary.
    """
    flat_d = {}
    for key, value in d.items():
        if isinstance(value, (dict, FrozenDict)):
            sub_dict = flatten_dict(value)
            for sub_key, sub_value in sub_dict.items():
                flat_d[key + separation + sub_key] = sub_value
        else:
            flat_d[key] = value
    return flat_d


def unflatten_dict(d: Dict, separation: str = "."):
    """Unflattens a dictionary, inverse to flatten_dict.

    Args:
        d (Dict): The dictionary to unflatten.
        separation (str, optional): The separation character. Defaults to ".".

    Returns:
        Dict: The unflattened dictionary.
    """
    unflat_d = {}
    for key, value in d.items():
        if separation in key:
            sub_keys = key.split(separation)
            sub_dict = unflat_d
            for sub_key in sub_keys[:-1]:
                if sub_key not in sub_dict:
                    sub_dict[sub_key] = {}
                sub_dict = sub_dict[sub_key]
            sub_dict[sub_keys[-1]] = value
        else:
            unflat_d[key] = value
    return unflat_d


def flatten_params(params: Any, num_batch_dims: int = 0):
    """Flattens the parameters of the model.

    Args:
        params (jax.PyTree): The parameters of the model.
        num_batch_dims (int, optional): The number of batch dimensions. Tensors will not be flattened over these dimensions. Defaults to 0.

    Returns:
        List[Tuple[str, List[int]]]: Structure of the flattened parameters.
        jnp.ndarray: The flattened parameters.
    """
    flat_params = flatten_dict(params)
    keys = sorted(list(flat_params.keys()))
    param_config = [(k, flat_params[k].shape[num_batch_dims:]) for k in keys]
    comb_params = jnp.concatenate(
        [flat_params[k].reshape(*flat_params[k].shape[:num_batch_dims], -1) for k in keys], axis=-1
    )
    return param_config, comb_params


def unflatten_params(
    param_config: List[Tuple[str, List[int]]],
    comb_params: jnp.ndarray,
):
    """Unflattens the parameters of the model.

    Args:
        param_config (List[Tuple[str, List[int]]]): Structure of the flattened parameters.
        comb_params (jnp.ndarray): The flattened parameters.

    Returns:
        jax.PyTree: The parameters of the model.
    """
    params = []
    key_dict = {}
    idx = 0
    for key, shape in param_config:
        params.append(
            comb_params[..., idx : idx + np.prod(shape)].reshape(*comb_params.shape[:-1], *shape)
        )
        key_dict[key] = 0
        idx += np.prod(shape)
    key_dict = unflatten_dict(key_dict)
    return FrozenDict(jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(key_dict), params))


#############################################
# Jraph utils
#############################################
def get_nef_graph_adjacency(
    nef_params: dict,
    nef_name: str,
):
    """Converts a NEF model to an adjacency matrix graph representation.

    Args:
        nef_params (dict): The flattened parameter dict of the NEF model. This contains e.g.
            {'kernel_net_0.linear_0.kernel': [batch_size, input_dim, hidden_dim], ...}
        nef_name (str): The type of NEF model. For now this can be one of "SIREN", "RFFNet", "MLP".
    """
    # MFNs require a different treatment, not implemented yet.
    assert nef_name in [
        "SIREN",
        "RFFNet",
        "MLP",
    ], "Only SIREN, RFFNet and MLP are currently supported."

    # Separate weights and biases
    weights = []
    biases = []
    for param_name, param in nef_params.items():
        if param_name.split(".")[-1] == "kernel":
            # weights[param_name] = nef_params[param_idx]
            weights.append(param)
        elif param_name.split(".")[-1] == "bias":
            # biases[param_name] = nef_params[param_idx]
            biases.append(param)
        else:
            raise ValueError("Unknown parameter name.")

    # Extract batch size
    batch_size = weights[0].shape[0]

    # Calculate the number of nodes per neural field computational graph.
    #   +   dimensionality of network input (2 dim coordinates)
    #   +   the number of activation units in each layer,
    #       as calculated by the number of output units in the previous layer.
    num_nodes = weights[0].shape[1] + sum(w.shape[2] for w in weights)

    # Create a node feature matrix. Size of this is equal to the number of nodes.
    node_features = np.zeros(shape=(batch_size, num_nodes, 1), dtype=np.float32)

    # Edge features are the weights of the network, 0 for non-connected nodes (no edges).
    edge_features = np.zeros(shape=(batch_size, num_nodes, num_nodes, 1), dtype=np.float32)

    # We fill the edge_feature matrix block-wise, every block corresponding to the weights of a layer.
    # row_offset : row_offset + num_in is a number of rows corresponding to the number of input neurons in a layer
    # col_offset : col_offset + num_out is a number of columns corresponding to the number of output neurons in a layer
    #       the values inserted at a given row_offset + i and range of col_offset : col_offset + num_out correspond to
    #       the weights of the layer that map from a given node row_offset + i to all output nodes.

    row_offset = 0  # Since the rows encode outgoing edges, we start at 0
    col_offset = weights[0].shape[1]  # Input nodes have no incoming edges.

    # Insert weights as edge features.
    for i, w in enumerate(weights):
        # Add feature dimension to weights
        w = jnp.expand_dims(w, -1)

        # Get number of input and output nodes for this layer
        _, num_in, num_out, _ = w.shape

        # Insert batch of weights into edge feature matrix
        edge_features[
            :, row_offset : row_offset + num_in, col_offset : col_offset + num_out, :
        ] = w

        # Increase row and column offset
        row_offset += num_in
        col_offset += num_out

    # Insert biases as node features.
    row_offset = weights[0].shape[1]  # no bias in input nodes
    for i, b in enumerate(biases):
        # Add feature dimension to biases
        b = jnp.expand_dims(b, -1)

        # Get number of output nodes for this layer
        _, num_out, _ = b.shape

        # Insert batch of biases into node feature matrix
        node_features[:, row_offset : row_offset + num_out, :] = b

        # Increase row offset
        row_offset += num_out

    # Convert to jax arrays
    node_features = jnp.array(node_features)
    edge_features = jnp.array(edge_features)

    return node_features, edge_features


# Define a named tuple for storing the nef graph structure of a single nef.
NefGraph = namedtuple("NefGraph", ["senders", "receivers", "n_node", "n_edge"])


def get_nef_graph_lists(
    nef_params: jnp.ndarray,
    nef_config: dict,
    nef_name: str,
    add_reverse_edges: bool = False,
):
    """Converts a NEF model to a graph tuple representation. (As used by Jraph)

    Args:
        nef_params (dict): The flattened parameter dict of the NEF model. This contains e.g.
            {'kernel_net_0.linear_0.kernel': [batch_size, input_dim, hidden_dim], ...}
        nef_name (str): The type of NEF model. For now this can be one of "SIREN", "RFFNet", "MLP".
    """

    # Graphs are represented by graph tuples which are aligned arrays of sender and receiver nodes.
    # The sender and receiver nodes are represented by their features and index in the node_features array.
    # MFNs require a different treatment, not implemented yet.
    assert nef_name in [
        "SIREN",
        "RFFNet",
        "MLP",
    ], "Only SIREN, RFFNet and MLP are currently supported."

    # Unflatten batch of weights to get a batch of nef params
    exmp_batch_nef_params = unflatten_params(nef_config, nef_params)

    # Flatten batch of nef params
    nef_params = flatten_dict(exmp_batch_nef_params)

    # Separate weights and biases
    weights = []
    biases = []
    for param_name, param in nef_params.items():
        if param_name.split(".")[-1] == "kernel":
            # weights[param_name] = nef_params[param_idx]
            weights.append(param)
        elif param_name.split(".")[-1] == "bias":
            # biases[param_name] = nef_params[param_idx]
            biases.append(param)
        else:
            raise ValueError("Unknown parameter name.")

    #################################################
    # Creating the sender and receiver nodes lists
    #################################################

    # Create two empty lists, these will contain the sender and receiver nodes that define the edges of the graph.
    sender_nodes = []
    receiver_nodes = []

    sender_offset = 0  # Since the rows encode outgoing edges, we start at 0
    receiver_offset = weights[0].shape[1]  # Input nodes have no incoming edges.
    for i, w in enumerate(weights):
        # Get number of input and output nodes for this layer
        _, num_in, num_out = w.shape

        # Create sender node list for this layer
        sender_nodes_layer = np.arange(sender_offset, sender_offset + num_in)
        # Repeat num_out times; [in_0, in_0, in_0, in_1, in_1, in_1]
        sender_nodes_layer = np.repeat(sender_nodes_layer, num_out)
        sender_nodes.append(sender_nodes_layer)

        # Create receiver node list for this layer
        receiver_nodes_layer = np.arange(receiver_offset, receiver_offset + num_out)
        # Tile num_in times; [out_0, out_1, out_2, out_0, out_1, out_2]
        receiver_nodes_layer = np.tile(receiver_nodes_layer, num_in)
        receiver_nodes.append(receiver_nodes_layer)

        # If the graph is undirected, we must add the reverse edges as well.
        if add_reverse_edges:
            sender_nodes.append(receiver_nodes_layer)
            receiver_nodes.append(sender_nodes_layer)

        # Increase row and column offset
        sender_offset += num_in
        receiver_offset += num_out

    # Create a single list of sender and receiver nodes for the entire graph of size [num_edges]
    sender_nodes = np.concatenate(sender_nodes, axis=None)
    receiver_nodes = np.concatenate(receiver_nodes, axis=None)
    num_edges_per_graph = len(sender_nodes)

    # Calculate the number of nodes per neural field computational graph.
    #   +   dimensionality of network input (2 dim coordinates)
    #   +   the number of activation units in each layer,
    #       as calculated by the number of output units in the previous layer.
    num_nodes = weights[0].shape[1] + sum(w.shape[2] for w in weights)

    # Store number of nodes and number of edges per graph.
    n_node = np.array([num_nodes])
    n_edge = np.array([num_edges_per_graph])

    # Create a nef graph structure
    nef_graph = NefGraph(sender_nodes, receiver_nodes, n_node, n_edge)

    # Return node and edge features, and the graph structure for a single nef graph.
    return nef_graph


def nefs_to_jraph_tuple(
    nef_params: jnp.ndarray,
    nef_config: ConfigDict,
    nef_graph: NefGraph,
    add_reverse_edges: bool = False,
):
    """Converts a batch of nef parameters to a jraph graph tuple.

    Args:
        nef_params (jnp.ndarray): The flattened weights of the nef model. This contains e.g. [batch_size, num_params]
        nef_config (ConfigDict): The nef config dict.
        nef_graph (NefGraph): The nef graph structure.
        add_reverse_edges (bool): Whether to add reverse edges to the graph. Defaults to False.
    """
    #############################################################
    # 1. Get node and edge features from nef params
    #############################################################

    # Unflatten set of weights to get a nested dict of nef params e.g.
    #   {'kernel_net_0': {'linear': {'kernel': [batch_size, input_dim, hidden_dim]}, ...}
    exmp_batch_nef_params = unflatten_params(nef_config, nef_params)

    # Flatten batch of nef params to get a flat dict of nef params e.g.
    #   {'kernel_net_0.linear_0.kernel': [batch_size, input_dim, hidden_dim], ...}
    nef_params = flatten_dict(exmp_batch_nef_params)

    nef_name = "SIREN"
    assert nef_name in [
        "SIREN",
        "RFFNet",
        "MLP",
    ], "Only SIREN, RFFNet and MLP are currently supported."

    # Separate weights and biases
    weights = []
    biases = []
    for param_name, param in nef_params.items():
        if param_name.split(".")[-1] == "kernel":
            weights.append(param)
        elif param_name.split(".")[-1] == "bias":
            biases.append(param)
        else:
            raise ValueError("Unknown parameter name.")

    batch_size = weights[0].shape[0]
    num_input_nodes = weights[0].shape[1]

    # Flatten weights from batch_dim, end up with list of weights of size [[batch_dim, in_channels*out_channels], ... ]
    flat_weights = [np.reshape(w, (batch_size, -1)) for w in weights]

    # Concatenate all weights together along the last dimension, get [batch_dim, in_channels*out_channels*layers]
    # Then reshape to [batch_dim, num_edges, 1]
    edge_features = np.concatenate(flat_weights, axis=-1)
    edge_features = np.expand_dims(edge_features, axis=-1)

    # If we use undirected graphs, we must add the reverse edges as well.
    if add_reverse_edges:
        edge_features = np.repeat(edge_features, 2, axis=1)

    node_features = np.concatenate((np.zeros((batch_size, num_input_nodes)), *biases), axis=-1)
    node_features = np.expand_dims(node_features, axis=-1)

    #############################################################
    # 2. Batch the node and edge features into a Jraph Tuple
    #############################################################

    # Unpack nef graph structure.
    sender_nodes = nef_graph.senders
    receiver_nodes = nef_graph.receivers
    n_node = nef_graph.n_node
    n_edge = nef_graph.n_edge

    # Get batch size.
    batch_size = node_features.shape[0]

    # Create a graph batch index list, this list contains, at position i, the node offset of the graph that node i
    # belongs to.
    batch_graph_index = np.arange(batch_size).repeat(n_edge)  # [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, ...]
    batch_graph_index = batch_graph_index * n_node  # [0, 0, 0, 0, 0, 0, 10, 10, 10, 10, ...]

    # Flatten node and edge features
    node_features = np.reshape(node_features, (-1, 1))  # [batch_size * n_node, 1]
    edge_features = np.reshape(edge_features, (-1, 1))  # [batch_size * n_edge, 1]

    # Tile sender and receiver nodes for batch size, then add the batch graph index to the nodes.
    sender_nodes = np.tile(sender_nodes, batch_size) + batch_graph_index
    receiver_nodes = np.tile(receiver_nodes, batch_size) + batch_graph_index

    # Repeat n_node and n_edge for batch size.
    n_node = np.repeat(n_node, batch_size)
    n_edge = np.repeat(n_edge, batch_size)

    # Create a jraph graph tuple.
    jraph_batch = jraph.GraphsTuple(
        n_node=n_node,
        n_edge=n_edge,
        nodes=node_features,
        edges=edge_features,
        globals=None,
        receivers=receiver_nodes,
        senders=sender_nodes,
    )
    return jraph_batch


def jraph_tuple_to_nefs(jraph_graph: jraph.GraphsTuple, nef_config: ConfigDict):
    """Converts a jraph graph tuple to a batch of nef parameters.

    Args:
        jraph_graph (jraph.GraphsTuple): The jraph graph to convert.
        nef_config (ConfigDict): The nef config dict.
    """
    pass


#############################################
# Visualisations
#############################################


def draw_jraph_graph_structure(jraph_graph: jraph.GraphsTuple):
    """Draws the structure of a jraph graph.

    Args:
        jraph_graph (jraph.GraphsTuple): The jraph graph to draw.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    nodes, edges, receivers, senders, _, _, _ = jraph_graph
    nx_graph = nx.DiGraph()
    if nodes is None:
        for n in range(jraph_graph.n_node[0]):
            nx_graph.add_node(n)
    else:
        for n in range(jraph_graph.n_node[0]):
            nx_graph.add_node(n, node_feature=nodes[n])
    if edges is None:
        for e in range(jraph_graph.n_edge[0]):
            nx_graph.add_edge(int(senders[e]), int(receivers[e]))
    else:
        for e in range(jraph_graph.n_edge[0]):
            nx_graph.add_edge(int(senders[e]), int(receivers[e]), edge_feature=edges[e])

    pos = nx.spring_layout(nx_graph)
    nx.draw(nx_graph, pos=pos, with_labels=True, node_size=500, font_color="yellow")
    plt.show()


def draw_jraph_multipartite_from_nefs(nef_cfg, nef_params):
    """Draws a multipartite graph from a list of nef parameters. Useful for inspecting whether i've
    implemented the graph structure correctly.

    Args:
        nef_cfg (dict): The nef config dict.
        nef_params (dict): The nef parameters dict.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    # Get the nef graph structure
    nef_graph = get_nef_graph_lists(nef_params=nef_params, nef_config=nef_cfg, nef_name="SIREN")
    # Unflatten set of weights to get a nested dict of nef params e.g.
    #   {'kernel_net_0': {'linear': {'kernel': [batch_size, input_dim, hidden_dim]}, ...}
    exmp_batch_nef_params = unflatten_params(nef_cfg, nef_params)

    # Flatten batch of nef params to get a flat dict of nef params e.g.
    #   {'kernel_net_0.linear_0.kernel': [batch_size, input_dim, hidden_dim], ...}
    nef_params = flatten_dict(exmp_batch_nef_params)

    # Create a jraph graph tuple
    jraph_graph = jraph.GraphsTuple(
        n_node=nef_graph.n_node,
        n_edge=nef_graph.n_edge,
        nodes=None,
        edges=None,
        globals=None,
        receivers=nef_graph.receivers,
        senders=nef_graph.senders,
    )

    # Separate weights and biases
    weights = []
    biases = []
    for param_name, param in nef_params.items():
        if param_name.split(".")[-1] == "kernel":
            # weights[param_name] = nef_params[param_idx]
            weights.append(param)
        elif param_name.split(".")[-1] == "bias":
            # biases[param_name] = nef_params[param_idx]
            biases.append(param)
        else:
            raise ValueError("Unknown parameter name.")

    nx_graph = nx.DiGraph()
    node_offset = 0

    # Loop over all weights in this nef
    for layer_idx, weight in enumerate(weights):
        # Get number of input nodes for this layer
        num_nodes_layer = weight.shape[1]

        for i in range(node_offset, node_offset + num_nodes_layer):
            if jraph_graph.nodes is None:
                nx_graph.add_node(i, layer=layer_idx)
            else:
                nx_graph.add_node(i, layer=layer_idx, node_feature=jraph_graph.nodes[i])

        node_offset += num_nodes_layer

    # Add nodes for output layer
    for i in range(node_offset, node_offset + weights[-1].shape[2]):
        if jraph_graph.nodes is None:
            nx_graph.add_node(i, layer=len(weights))
        else:
            nx_graph.add_node(i, layer=len(weights), node_feature=jraph_graph.nodes[i])

    # Add edges
    for i in range(jraph_graph.n_edge[0]):
        if jraph_graph.edges is None:
            nx_graph.add_edge(int(jraph_graph.senders[i]), int(jraph_graph.receivers[i]))
        else:
            nx_graph.add_edge(
                int(jraph_graph.senders[i]),
                int(jraph_graph.receivers[i]),
                edge_feature=jraph_graph.edges[i],
            )

    pos = nx.multipartite_layout(nx_graph, subset_key="layer")
    nx.draw(
        nx_graph, pos=pos, with_labels=True, node_size=500, node_color="yellow", font_color="black"
    )
    plt.show()


def draw_jraph_multipartite_from_nefs_batched(nef_cfg: ConfigDict, nef_params, batch_size=3):
    """Draws a multipartite graph from a list of nef parameters. Useful for inspecting whether i've
    implemented the graph structure correctly.

    Args:
        nef_cfg (dict): The nef config dict.
        nef_params (dict): The nef parameters dict.
        batch_size (int): The batch size to plot.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    # Get the nef graph structure
    nef_graph = get_nef_graph_lists(
        nef_config=nef_cfg,
        nef_params=nef_params,
        nef_name="SIREN",
    )

    # Create a batched jraph graph tuple
    jraph_graph = nefs_to_jraph_tuple(
        nef_params=nef_params,
        nef_config=nef_cfg,
        nef_graph=nef_graph,
    )

    # Separate weights and biases
    weights = []
    biases = []
    for param_name, param in nef_params.items():
        if param_name.split(".")[-1] == "kernel":
            # weights[param_name] = nef_params[param_idx]
            weights.append(param)
        elif param_name.split(".")[-1] == "bias":
            # biases[param_name] = nef_params[param_idx]
            biases.append(param)
        else:
            raise ValueError("Unknown parameter name.")

    nx_graph = nx.DiGraph()
    for graph_idx in range(batch_size):
        # Increase offset by number of nodes in all previous graphs
        node_offset = sum(jraph_graph.n_node[:graph_idx])

        # Set layer offset as sum of number of layers in all previous graphs
        layer_offset = graph_idx * (len(weights) + 1)

        # Loop over all weights in this nef
        for layer_idx, weight in enumerate(weights):
            # Get number of input nodes for this layer
            num_nodes_layer = weight.shape[1]

            for i in range(node_offset, node_offset + num_nodes_layer):
                if jraph_graph.nodes is None:
                    nx_graph.add_node(i, layer=layer_offset + layer_idx)
                else:
                    nx_graph.add_node(
                        i, layer=layer_offset + layer_idx, node_feature=jraph_graph.nodes[i]
                    )

            node_offset += num_nodes_layer

        # Add nodes for output layer
        for i in range(node_offset, node_offset + weights[-1].shape[2]):
            if jraph_graph.nodes is None:
                nx_graph.add_node(i, layer=layer_offset + len(weights))
            else:
                nx_graph.add_node(
                    i, layer=layer_offset + len(weights), node_feature=jraph_graph.nodes[i]
                )

        # Increase offset by number of edges in all previous graphs
        edge_offset = sum(jraph_graph.n_edge[:graph_idx])

        # Add edges
        for i in range(jraph_graph.n_edge[graph_idx]):
            if jraph_graph.edges is None:
                nx_graph.add_edge(
                    int(jraph_graph.senders[edge_offset + i]),
                    int(jraph_graph.receivers[edge_offset + i]),
                )
            else:
                nx_graph.add_edge(
                    int(jraph_graph.senders[edge_offset + i]),
                    int(jraph_graph.receivers[edge_offset + i]),
                    edge_feature=jraph_graph.edges[edge_offset + i],
                )

    pos = nx.multipartite_layout(nx_graph, subset_key="layer")
    nx.draw(
        nx_graph, pos=pos, with_labels=True, node_size=500, node_color="yellow", font_color="black"
    )
    plt.show()
