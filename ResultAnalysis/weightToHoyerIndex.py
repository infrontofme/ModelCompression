def weightToHoyer(weight, sess, group=1):
    """
    calculate the sparsity of weight at group level using Hoyer Index.
    :param weight: 4D-Tensor, with shape = [kernel_size, kernel_size, in_channel, out_channel].
    :param group: int, how many groups you divided, default = 1.
    :param sess: tf.Session(), used to run tensorflow session.
    """
    assert weight.get_shape()[0] == 1, "The kernel size must be 1x1"
    assert weight.get_shape()[1] == 1, "The kernel size must be 1x1"
    
    # squeeze to 2D Tensor
    weight = tf.squeeze(weight) #[in_channel, out_channel]
    
    # convert to numpy array
    weight = sess.run(weight)
    
    # get size
    col_size = weight.shape[1]
    row_size = weight.shape[0]
    ele_per_group = col_size // group
    assert col_size % group == 0, "group number cannot be divided by output channels"
    HoyerIndex = []
    
    # calculate HoyerIndex by group
    for i in range(0, group):
        
        # weight contributions to each group, evaluate how important the weight is.
        group_l1_norm = np.linalg.norm(weight[:, i*ele_per_group:(i+1)*ele_per_group], ord=1, axis=1, keepdims=True)
        
        # l1-norm by col
        l1_norm = np.linalg.norm(group_l1_norm, ord=1, axis=0, keepdims=True)

        # l2-norm by col
        l2_norm = np.linalg.norm(group_l1_norm, ord=2, axis=0, keepdims=True)

        #calculate Hoyer Index of each group
        """
        (sqrt(size) - l1_norm/l2_norm)/(sqrt(size) - 1)

        0- least sparse
        1- most sparse
        """
        hoyer = (math.sqrt(row_size) - l1_norm[0, 0]/max(l2_norm[0, 0], 1e-10))/(math.sqrt(row_size) - 1)
        if hoyer > 1:
            hoyer = 1
        HoyerIndex.append(hoyer)

    return HoyerIndex