## 📌 注意事項

1. **環境需求**  
   - 需安裝 Python 3.8 以上(可能)
   - 需安裝 Pytorch 及 Numpy  

2. **可能會發生問題**  
   - 目前只有測試Pytorch `conv` 以及 `FC` 兩個層的Weight提取、轉換。其他並未實裝。 
   - 如果讀完發現還是不會用，請洽ChatGpt，真香。  

  
## 🚀 安裝與使用
### 1️⃣ 執行

```sh
py gen_params [model_file_name] [constant]
```

### 2️⃣ 一點點介紹

這個Python基本上就是能將model中的參數提取出來，並透過純整數計算的方式，獲取所有需要的參數。目前我們只支持到conv以及fc的參數提取，如果有需要其他的layer，可能就需要多加一些程式碼上去。

目前程式可讀性不太高，可以想成就是用硬幹的方式實現。彈性不高，由於我沒時間搞因此就先暫時這樣。

下面的公式要好好看，以利於使用參數進行推導。




### Quantize formula
```math
$Q_{out} = clmap(round(\dfrac{x_{input}}{scale}+zeropoint),Q_{min},Q_{max})$
```
一開始在模型中的輸入必須先轉換為int8或者uint8形式，因此以上公式為tensorflow純粹將tensor轉換為quant的方式。

#### Conv quant formula
```math
$Q_{conv2D}=clamp(round(\dfrac{S_wS_{input}}{S_{conv2D}}\sum{(Q_{input}-zp_{input})*(Q_w-zp_w)}+zp_{conv2D}),Q_{min},Q_{max})$
```
模型中conv2D的公式為
```math
$y_{f} = \sum{x_f}{w_f}+y_{bias}$
```
而在硬體架構中我們的輸入及權重都是在-128~127 or 0~255當中

因此我們便須將輸入都改成為Quantize的版本

$S_3(y_q-y_{zp})= \sum{S_1}(x_q-x_{zp})S_2(w_q-w_{zp}) +y_{bias}$

$y_q-y_{zp} = \dfrac{S_1S_2}{S_3}\sum(x_q-x_{zp})(w_q-w_{zp})+\dfrac{y_{bias}}{S_3}$

我們將 $M =\dfrac{S_1S_2}{S_3}$

則：

$(y_q-y_{zp})=M\sum{(x_q-x_{zp})(w_q-w_{zp})}+M\dfrac{y_{bias}}{S_1S_2}$

$(y_q-y_{zp}) = M\{\sum{(x_q-x_{zp})({w_q-w_{zp})+\dfrac{y_{bias}}{S_1S_2}}}\}$


$y_q = M\{\sum{(x_q-x_{zp})({w_q-w_{zp})+\dfrac{y_{bias}}{S_1S_2}}}\}+y_{zp}$

而到這邊我們就推導出來正確的公式。

而在Pytorch及Tensorflow當中都必須要四捨五入後做夾擠

$output = clamp(round(y_q,Q_{min},Q_{max}))$

我們可以定義multiplier $M\approx\dfrac{x_{scale}w_{scale}}{scale}$

而 $M = 2^{-n}M_0$  

n : 為非零整數

$M_0:$ a fixed-point multiplier

$M :$ floating point 

ref:
[Quantization and Training of Neural Networks for Efficient
Integer-Arithmetic-Only Inference
](https://arxiv.org/pdf/1712.05877)
#### Conv forumla in Hardware
$y_q = M\{\sum{(x_q-x_{zp})({w_q-w_{zp})+\dfrac{y_{bias}}{S_1S_2}}}\}+y_{zp}$

$y_f=M(\sum{x_qw_q}-w_{zp}\sum{x_q}-x_{zp}\sum{w_q}+\sum{x_{zp}w_{zp}}+\dfrac{y_{bias}}{S_1S_2})+y_{zp}$

**由於我們在Pytorch中訓練QAT時，在x86的預設下，Weight會是以Symmertic的方式去訓練，因此不存在zero points。公式我們可以直接改寫成**


$a_2 = \sum{w_q}$ 

$y_f=Z_3+2^{-n}M_0(\sum{x_qw_q}+Z_1a_2+bias)$


