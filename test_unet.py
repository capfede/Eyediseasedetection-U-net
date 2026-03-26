from unet_predict import predict_dr

result, mask = predict_dr("static/images/test.jpg")

print(result)
print(mask)