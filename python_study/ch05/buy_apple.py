import MulLayer as ml

apple = 100
apple_num = 2
tax = 1.1

#Layer
mul_apple_layer = ml.MulLayer()
mul_tax_layer = ml.MulLayer()

#forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)


# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple_price, dtax, dapple, dapple_num)