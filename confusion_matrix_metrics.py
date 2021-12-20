
def convert_prob(probs):
    converted_probs = []
    for i in range(len(probs)):
        if(probs[i] > 0.5):
            converted_probs.append(1.)
        else:
            converted_probs.append(0)   
    return converted_probs          

def conf_mat(prob_arr, input_arr):

        # confusion matrix
        conf_arr = [[0, 0], [0, 0]]
        n=len(input_arr)
        for i in range(n):
                if int(prob_arr[i]) == 0 :
                        if float(prob_arr[i]) < 0.5:
                                conf_arr[0][0] = conf_arr[0][0] + 1
                        else:
                                conf_arr[0][1] = conf_arr[0][1] + 1
                elif int(input_arr[i]) == 1:
                        if float(prob_arr[i]) >= 0.5:
                                conf_arr[1][1] = conf_arr[1][1] +1
                        else:
                                conf_arr[1][0] = conf_arr[1][0] +1

        return conf_arr
        
def metrics(conf_arr,input_arr):
    accuracy=float(conf_arr[0][0] + conf_arr[1][1])/(len(input_arr))
    senzitivity=float (conf_arr[1][1]/(conf_arr[1][1]+conf_arr[1][0]))
    specifity=float(conf_arr[0][0]/(conf_arr[0][0]+conf_arr[0][1]))
    precision=float(conf_arr[1][1]/(conf_arr[1][1]+conf_arr[0][1]))
    FPR=float(conf_arr[0][1]/(conf_arr[0][0]+conf_arr[0][1])) 
    f1=2*(precision*senzitivity/(precision+senzitivity))
    return accuracy,senzitivity,specifity,precision,FPR,f1
