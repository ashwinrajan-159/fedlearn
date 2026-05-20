import onnx
from onnx import helper
from onnx import TensorProto

def create_model(output_path="host/model.onnx"):
    # Define I/O tensors
    # X: Input features [batch_size, 4]
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, ['batch_size', 4])
    
    # W: Weights [4, 3]
    W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [4, 3])
    
    # B: Bias [3]
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [3])
    
    # Y: Output probabilities [batch_size, 3]
    probs = helper.make_tensor_value_info('probs', TensorProto.FLOAT, ['batch_size', 3])
    
    # Define nodes
    # 1. MatMul: X @ W -> logits_no_bias
    node_matmul = helper.make_node(
        'MatMul',
        inputs=['X', 'W'],
        outputs=['logits_no_bias'],
        name='matmul_node'
    )
    
    # 2. Add: logits_no_bias + B -> logits
    node_add = helper.make_node(
        'Add',
        inputs=['logits_no_bias', 'B'],
        outputs=['logits'],
        name='add_node'
    )
    
    # 3. Softmax: Softmax(logits) -> probs
    node_softmax = helper.make_node(
        'Softmax',
        inputs=['logits'],
        outputs=['probs'],
        axis=1,
        name='softmax_node'
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[node_matmul, node_add, node_softmax],
        name='LinearRegressionSoftmax',
        inputs=[X, W, B],
        outputs=[probs]
    )
    
    # Create the model
    model = helper.make_model(graph, producer_name='fedlearn-host')
    
    # Define opset import (ensure compatibility with web worker)
    opset = model.opset_import.add()
    opset.domain = ''
    opset.version = 13 # Opset 13 is widely supported in onnxruntime-web
    
    # Save the model
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Model successfully saved to {output_path}")

if __name__ == "__main__":
    create_model()
