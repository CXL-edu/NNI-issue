```python
"""
在使用pytorch定义网络结构时，我们需要继承nn.Module的功能
*如果使用python2的继承方式super(Class_name, self)则会报错，使用python3的继承方式super()可以*
然而许多代码中这两种方式通常是兼容的，因此需要官方做兼容处理或者在快速开始文档写入Note

*实例化对象时，传入参数也有许多注意事项*
直接传入数值型数据、元组、列表和字典是没有问题的。传入argparse生成的对象也没有问题。
好吧，这不算一个问题，只是传入我写的DotDict对象时有问题，但是增加__getstate__和__setstate__方法之后解决了该问题，
使用DotDict对象的初衷是想对嵌套字典都可以直接使用属性找到对应的值，如model.mlp.hidden_size。
在NNI库中不兼容的原因应该是，其在decorator中将传入参数使用copy.copy方法拷贝导致的
这只是一个小的兼容问题，大多数人应该不会自定义类似的参数对象
"""

# 在实例化模型时，如果不需要传入参数会报如下错误
RecursionError: maximum recursion depth exceeded while calling a Python object
# 在实例化模型时，如果需要传入参数会报如下错误
TypeError: Net.__init__() missing 1 required positional argument: 'configs'
        
# 回到第一个问题，下面的写法会导致错误，不论是否传参
@model_wrapper      # this decorator should be put on the out most
class Model(nn.Module):
    def __init__(self, configs):
        super(Net, self).__init__()
        
# 这样是正确的，python3的继承写法
@model_wrapper      # this decorator should be put on the out most
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
```

