---
layout: page
title: "Get the length of std::array without instantiate an object."
teaser: "A little bit of Template Meta Programming"
header: no
image:
    title: c++.png
    thumb:  c++.png
    caption: std::array<T,N> 
comments: true
categories:
    - c++
---

### Get the container length of std::array

You have declared an object with type std::array:

```cpp
std::array<int, 3> data = {1,2,3}; 
```

To get the length of it is trivial:

```cpp
auto length = data.size(); 
```

However, can you get the length of it just from the type? Imagine in your code base an array type is declared using an alias:

```cpp
//in samples.h
using Samples = std::array<int, 3>;
```

Only the *samples.h* is visible to you and you want to declare another array with different element type but same size as Samples. You don't want to hard-code another 3, cuz that reduces code reusability.
You could do the following, but it is ugly:

```cpp
Samples dummy{};
using FloatSamples = std::array<float, dummy.size()>;
```

### Solution
Here I show a nicer way using some C++ template magic.

First we declare an template class which can take the std::array type as template argument: 

```cpp
//general case
template <typename Container>
struct helper
{
};
```

Then we add a specialization of the above template class. On the one hand, it let std::array type fall into this specialization category. On the other hand, we have an opportunity to extract the element type T and the array size N:

```cpp 
//specialization on array
template <typename T, size_t N>
struct helper<std::array<T, N>>
{
	static constexpr std::size_t value{N};
};
```

If we put the above helper code into *helper.h*, and put everything together, then we have:

```cpp
#include "samples.h"
#include "helper.h"
int main()
{
	using FloatSamples = std::array<float, helper<Samples>::value>;
	std::cout << helper<Samples>::value<< std::endl; // will print 3
	return 0;
}
```

{% if true %}
<div class="ads">
{% include advertising_wide.html %}
</div><!-- /.ads -->
{% endif %}


### Summary

In the end, we are able to just use std::array type to induce the size of it at compile time, without a need to instantiate an instance of it. Thanks to the magic of template metaprogramming.
The trick here we are using has a formal terminology, and is called *meta function*. 

For more on meta function, I have a 10 min [video]({{ site.url }}design/metaprogramming-vid) explaining it in detail!

[1]: https://docs.google.com/presentation/d/e/2PACX-1vQwrivdqqBR8teLQ7prKtiDyMLSqgGBzTxfQ6BKXPVvpFpLRUQOmqTm57LEMIy3IIK14RTLcBcT-PCO/pub?start=false&loop=false&delayms=60000&slide=id.g1a727d4a2c_0_814
[2]: https://i.stack.imgur.com/lV7Ty.jpg
[3]: http://machinelearningmechanic.com/deep_learning/2019/09/04/cross-entropy-loss-derivative.html
[4]: https://www.tensorflow.org/tutorials/quickstart/beginner
[5]: https://i.stack.imgur.com/uGw1c.jpg

