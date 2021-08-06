def predictor(gru: RNN, data: pd.DataFrame, preds: int):
    cell = gru.gru
    lin = gru.l1
  

    known = torch.from_numpy(data.iloc[:,0].to_numpy()).cuda().double()
    hid = torch.zeros((1,256)).cuda().double()
    lval=0
    for k,val in enumerate(known):
        val=val.unsqueeze(0)
        lval=val
        hid = cell(val,hid)
  
    pred = np.empty(preds)
    for i in range(preds):
        hid = cell(lval,hid)
        lval = lin(relu(hid))
        pred[i] = float(lval[0])

    return pred

    
