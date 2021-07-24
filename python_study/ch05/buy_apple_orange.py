import MulLayer as ml
import AddLayer as al

apple = 100
apple_num  = 2
orange = 150
orange_num = 3
tax = 1.1

#Layer
mul_apple_layer = ml.MulLayer()
mul_orange_layer = ml.MulLayer()
add_apple_orange_layer = al.AddLayer()
mul_tax_layer = ml.MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)
print(price)

#backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(price)
print(dapple, dapple_num, dorange, dorange_num, dtax)