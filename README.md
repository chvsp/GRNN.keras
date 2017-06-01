# Keras Implementation of Grounded RNNs 

This is a implementation of [Grounded RNNs](https://arxiv.org/abs/1705.08557) as proposed by Ankit Vani, et. al

I have modified the GRU and Dense Layers of Keras into Grounded_GRU and Grounded_Dense respectively to add the functionality mentioned in the paper.

The main modifications in these layers are as follows:

 * GRU
 	* Multiplied the recurrent weight matrices, which correspond to the label states of the hidden vector, with an identity matrix to convert all non-diagonal elements to zero.
 	* WARNING: My implementation only supports GRU models with `implementation` parameter set to `2`
 * Dense
 	* Performs an inverse sigmoid operation on the grounded vector and then applies the affine transformation. For more details refer to 


#### TODO
* Add sample implementation for sequence classification.
