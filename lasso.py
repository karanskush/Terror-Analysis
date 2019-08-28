


ENet.fit(final_data, train_y)
KRR.fit(final_data, train_y)

predicted_train = ENet.predict(x_test1)
predicted_train1 = KRR.predict(x_test1)

predicted_train = predicted_train.reshape(2000, 1)
predicted_train1 = predicted_train1.reshape(2000, 1)

new = scaler.inverse_transform(predicted_train) - y_test
new1 = scaler.inverse_transform(predicted_train1) - y_test

print(np.sum(new ** 2) / 2000)
print(np.sum(new1 ** 2) / 2000)

new_int = np.around(scaler.inverse_transform(predicted_train), decimals=0)

predicted_train = scaler.inverse_transform(predicted_train)

averaged_models = StackingAveragedModels(base_models=(ENet, rf), meta_model=KRR)
averaged_models.fit(final_data, train_y)
predicted_train = averaged_models.predict(x_test1)

predicted_train = predicted_train.reshape(2000, 1)

new_int = np.around(scaler.inverse_transform(predicted_train), decimals=0)

new = scaler.inverse_transform(predicted_train) - y_test

print(np.sum(new ** 2) / 2000)

df = pd.DataFrame(data=scaler.inverse_transform(predicted_train), columns=['predicted'])
df1 = pd.DataFrame(data=y_test, columns=['Real'])
result = pd.concat([df, df1], axis=1)

result.to_csv("show1.csv")
