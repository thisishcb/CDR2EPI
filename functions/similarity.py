def similarity_point_calculator(Prediction,ground_truth):
    Prediction=Prediction.replace(">","").replace("<","").replace(".","")
    similarity_point=0
    max_similarity_point=0
    for i in range(len(ground_truth)):
        pred_i=Prediction[i]
        true_i=ground_truth[i]
        similarity_point+=blosum50[pred_i][blosum50_keys.index(true_i)]
        max_similarity_point+=blosum50[true_i][blosum50_keys.index(true_i)]
        print(pred_i,true_i,blosum50[pred_i][blosum50_keys.index(true_i)])
    similarity_ratio=similarity_point/max_similarity_point
    return(similarity_point,max_similarity_point,similarity_ratio)
