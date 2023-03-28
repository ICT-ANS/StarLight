import torch
from torch.autograd import Variable
import torch.nn as nn

from engine.file_utils import save_layer_img

# A simple hook class that returns the input and output of a layer during forward pass
class Hook():
    def __init__(self, module, backward=False):
        uid = id(module)
        self.layer_id = "None"

        if hasattr(module, "weight"):
            # module.register_buffer("layer_id", torch.tensor(layer_id, dtype=torch.float64, requires_grad=True))
            layer_id = torch.nn.Parameter(torch.tensor(uid, dtype=torch.float64))
            module.register_parameter("layer_id", layer_id) #"layer_id" will appears in named_parameters()
            self.layer_id = str(uid)

        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output.cpu().detach().numpy()
        self.layer_name = module._get_name()

    def close(self):
        self.hook.remove()

# def register_hook(model):
#     seen = []
#     hook_list = []

#     for name, layer in list(model.named_modules()):
#         if hasattr(layer, "weight"):
#             ids = str(id(layer.weight)) + "-" + layer._get_name()
#             if ids not in seen:
#                 seen.append(ids)
#                 hook_list.append(Hook(layer))
#             else:
#                 print("duplicate")     
#     return hook_list

def register_hook(model):
    seen = []
    hook_list = []

    def traversal_layers(layer):
        layer_children = list(layer.children())
        for i in range(len(layer_children)):
            if hasattr(layer_children[i], "weight"):
            # if type(layer_children[i]) == nn.Conv2d:
                ids = str(id(layer_children[i].weight)) + "-" + layer_children[i]._get_name()
                if ids not in seen:
                    seen.append(ids)
                    hook_list.append(Hook(layer_children[i]))
                else:
                    print("duplicate")

            elif type(layer_children[i]) == nn.Sequential:
                for j in range(len(layer_children[i])):
                    tmp_childs = list(layer_children[i][j].children())
                    if len(tmp_childs) >0:
                        traversal_layers(layer_children[i][j])
                    else:
                        child = layer_children[i][j]
                        if hasattr(child, "weight"):
                            ids = str(id(child.weight)) + "-" + child._get_name()
                            if ids not in seen:
                                seen.append(ids)
                                hook_list.append(Hook(child))
                            else:
                                print("duplicate")
            else:
                traversal_layers(layer_children[i])
                
    traversal_layers(model)    

    return hook_list


def make_dot(var, params=None):
    """ Produces representation of PyTorch autograd graph.
    ref: https://github.com/szagoruyko/pytorchviz/blob/master/torchviz/dot.py
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """

    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    raw_graph = {"nodes":[], "edges":[]}

    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    output_nodes = (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)

    def add_nodes(var):
        if var not in seen:     
            if torch.is_tensor(var):
                # note: this used to show .saved_tensors in pytorch0.2, but stopped
                # working as it was moved to ATen and Variable-Tensor merged
                onenode = {"id":str(id(var)), "size":size_to_str(var.size()), "class_name":"Unknown", "name":str(id(var))}
                raw_graph["nodes"].append(onenode)

            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''

                layer_id = "None"
                if ".weight" in name:
                    prefix = name.replace("weight", "layer_id")
                    if prefix in params:
                        layer_id =  str(int(params[prefix].item()))
                        # print (layer_id, name)

                onenode = {"id":str(id(var)), "layer_id":layer_id, "size":size_to_str(u.size()), "class_name":"variable", "name":name}
                raw_graph["nodes"].append(onenode)

            elif var in output_nodes:
                layer_id = "None"
                class_name = str(type(var).__name__).replace("Backward","")
                onenode = {"id":str(id(var)), "layer_id":layer_id, "size":"None", "class_name":class_name, "name":str(id(var))}
                raw_graph["nodes"].append(onenode)

            else:
                layer_id = "None"

                class_name = str(type(var).__name__).replace("Backward","")
                onenode = {"id":str(id(var)), "layer_id":layer_id,  "size":"None", "class_name":class_name, "name":str(id(var))}
                raw_graph["nodes"].append(onenode)

            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        # dot.edge(str(id(u[0])), str(id(var)))
                        oneedge = {"source":str(id(u[0])), "target":str(id(var))}
                        raw_graph["edges"].append(oneedge)

                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    oneedge = {"source":str(id(t)), "target":str(id(var))}
                    raw_graph["edges"].append(oneedge)
                    add_nodes(t)

    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)
    json_graph = prune_graph(raw_graph)
    return json_graph

def prune_graph(graph):

    black_node_list = []
    white_node_list = []
    #record bias and weights node which are "variable" 
    for i, node in enumerate(graph["nodes"]) :
        if node["class_name"] == "variable":
            black_node_list.append(node)
        else:
            white_node_list.append(node)
    
    white_edge_list = []
    node_name_cache = {}
    
    #process node that is trainable (Conv2D BatchNorm) and update "name" props
    for i, edge in enumerate(graph["edges"]) :
        flag = True
        for idx, black_node in enumerate(black_node_list):
            source_id = edge["source"]
            target_id = edge["target"]

            if source_id == black_node["id"]:
                flag = False
                if target_id not in node_name_cache:
                    node_name_cache[target_id] = {}

                    new_name = black_node["name"].replace(".bias","").replace(".weight", "")
                    node_name_cache[target_id]["name"] = new_name

                if ".bias" in black_node["name"]:
                    node_name_cache[target_id]["bias"] = black_node["size"]
 
                if ".weight" in black_node["name"]:
                    node_name_cache[target_id]["weight"] = black_node["size"]
                    node_name_cache[target_id]["layer_id"] = black_node["layer_id"]
        if flag:
            white_edge_list.append(edge)


    # update node name and config
    for i, node in enumerate(white_node_list):
        id = node["id"]
        if id in node_name_cache:
            white_node_list[i]["name"] = node_name_cache[id]["name"]
            if "bias" in node_name_cache[id]:
                if white_node_list[i]["size"] == "None":
                    white_node_list[i]["size"] = {}
                white_node_list[i]["size"]["bias"] = node_name_cache[id]["bias"]
            if "weight" in node_name_cache[id]:
                if white_node_list[i]["size"] == "None":
                    white_node_list[i]["size"] = {}
                white_node_list[i]["size"]["weight"] = node_name_cache[id]["weight"]
            
            if "layer_id" in node_name_cache[id] and white_node_list[i]["layer_id"]=="None":
                white_node_list[i]["layer_id"] = node_name_cache[id]["layer_id"]
              

    #update node name again, "name" props should be unique
    used_name = []
    for i, node in enumerate(white_node_list):
        name = node["name"]

        if name in used_name: #avoid duplicated name
            name += "1"
            white_node_list[i]["name"] = name 
        used_name.append(name)

        if name.isdigit(): #avoid digital name
            #update node name
            new_name = "{}-{}".format(name, node["class_name"])
            white_node_list[i]["name"] = new_name
            #update edge name
            for k, edge in enumerate(white_edge_list):
                if name in edge["source"]:
                    white_edge_list[k]["source"] = new_name
                if name in edge["target"]:
                    white_edge_list[k]["target"] = new_name
        else: #
            id = node["id"]
            for k, edge in enumerate(white_edge_list):
                if id in edge["source"]:
                    white_edge_list[k]["source"] = node["name"]
                if id in edge["target"]:
                    white_edge_list[k]["target"] = node["name"] 

    
    # rebuild graph
    new_graph = {}
    for node in white_node_list:
        name = node["name"]
        if name not in new_graph:
            new_graph[name] = {"class_name": "", "inbound_nodes": []}

        new_graph[name]["class_name"] = node["class_name"]
        if node["size"] != "None":
            node["size"]["layer_id"] = node["layer_id"]

        new_graph[name]["config"] = node["size"]

    for edge in white_edge_list:
        source_id = edge["source"]
        target_id = edge["target"]

        if len(new_graph[target_id]["inbound_nodes"])==0 :
            new_graph[target_id]["inbound_nodes"].append([[source_id]])
        else:
            new_graph[target_id]["inbound_nodes"][0].append([source_id])


    json_graph = {"class_name": "Model", "config": {"name": "vgg16", "layers": []}}
    for key in new_graph:

        class_name = new_graph[key]["class_name"]
        if "Convolution" in new_graph[key]["class_name"]:
            class_name = "Convolution2D"
        
        name = key

        onenode = {"class_name": class_name, "config": new_graph[key]["config"], \
                "name": name, "inbound_nodes": new_graph[key]["inbound_nodes"]}
        json_graph["config"]["layers"].append(onenode)
    
    import json
    js = json.dumps(json_graph, sort_keys=False, indent=4, separators=(',', ': '))
    with open("model.json", "w") as f:
        f.writelines(js)
    
    return json_graph