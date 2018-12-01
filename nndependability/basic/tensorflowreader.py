import numpy as np
import tensorflow as tf

from . import neuralnet

def loadMlpFromTensorFlow(tfModelPath, layersNamesToBeExtracted, layerTypes, inputShape = (480, 640, 3), inputDataType = np.uint8, inputModelPath = "import/input:0"):
    """ Extract a Tensorflow model its MLP part to form a network suitable for formal analysis, by running the model. 

        Keyword arguments:
        model_path -- path of the model in .pb format
        input_shape -- input of the network. Internally it will be added with another dimension 1 for executing the batch of size 1.  
        inputModelPath - how the input should be fed (e.g, "import/input:0")
        layersNamesToBeExtracted -- sequence of layers where the engine should follow and extract. Non-empty
        layerTypes -- for the layers, what are their type. Currently it can be relu, linear, BN-gamma1 (BN with gamma set to default), BN, elu
        
    """

    nNet = neuralnet.NeuralNetwork()
    
    
    # Load GraphDef and Graph
    with tf.gfile.FastGFile(tfModelPath, "rb") as f:
        graphDef = tf.GraphDef()
        graphDef.ParseFromString(f.read())
    graph = tf.Graph()

    with graph.as_default():
        tf.import_graph_def(graphDef)
        
    inputToNetwork = np.zeros(inputShape, dtype=inputDataType)  
    # Create a batch of size 1
    inputToNetwork = np.expand_dims(inputToNetwork, axis=0)    
        
    if len(layersNamesToBeExtracted) == 0 or (len(layersNamesToBeExtracted)!= len(layerTypes)) :
        raise ValueError
        
    else:
    
        with tf.Session(graph=graph) as sess:
            for layerIndex in range(len(layersNamesToBeExtracted)):

                print("Processing layer "+layersNamesToBeExtracted[layerIndex])
                if layerTypes[layerIndex] != "BN" and layerTypes[layerIndex] != "BN-gamma1":
                    tfWeights = graph.get_tensor_by_name(layersNamesToBeExtracted[layerIndex] + "/weights:0")
                    tfBiases = graph.get_tensor_by_name(layersNamesToBeExtracted[layerIndex] + "/biases:0")
                
                    weights, bias = sess.run([tfWeights, tfBiases], feed_dict={inputModelPath: inputToNetwork})
                    # IMPORTANT: weights should be swapped axis, as the formulation in tensor flow and pytorch is different 
                    weights = np.swapaxes(weights,0,1)
                    
                    if layerTypes[layerIndex] == "linear":
                        nNet.addLinearLayer(weights, bias)
                    elif  layerTypes[layerIndex] == "relu":  
                        nNet.addReLULayer(weights, bias)
                    elif  layerTypes[layerIndex] == "elu":  
                             # gamma is by default 1.
                        alpha = np.ones(bias.shape)  
                        nNet.addELULayer(weights, bias, alpha)
                    else:
                        raise Exception("Unknown layer type")
                        
                elif layerTypes[layerIndex] == "BN-gamma1":  
                    tfMovingMean = graph.get_tensor_by_name(layersNamesToBeExtracted[layerIndex] + "/moving_mean:0")  
                    tfMovingVariance = graph.get_tensor_by_name(layersNamesToBeExtracted[layerIndex] + "/moving_variance:0")                  
                    tfBeta = graph.get_tensor_by_name(layersNamesToBeExtracted[layerIndex] + "/beta:0")   
                    movingMean, movingVariance, beta = sess.run([tfMovingMean, tfMovingVariance, tfBeta], feed_dict={inputModelPath: inputToNetwork})

                    # gamma is by default 1.
                    gamma = np.ones(beta.shape)    
                    
                    nNet.addBatchNormLayer(movingMean, movingVariance, gamma, beta)
                    
                else:
                    raise NotImplementedError("Encountered BN with gamma != 1")
                    
        nNet.checkConsistency()
        return nNet


        
    