### 1. 개요
   1. 문자열 관련, 특히나 바로 이전에 이메일 파일을 HTML파일로 변경하고 어간 추출, 이메일 헤더 제거, 소문자 변환, 구두점 제거, 특성 벡터로 바꿔주기 등의 과정을 거쳐야 했었는데 그 과정에서 내가 모르는 부분이 너무 많다는 것을 깨달았다.
   2. 이미지 전처리를 하는 과정도 꽤 어려워 했었지만 ImageDataGenerator을 이용하고 이미지 데이터의 shape가 2차원의 경우 (전체 사례의 개수, 특징 픽셀, 특징 픽셀), 1차원의 경우 (전체 사례 수, 특징픽셀x특징픽셀)이런 느낌이라는 것을 깨닫고 한결 편해졌으며  
   csv파일이나 txt파일의 경우 읽기가 매우 쉬웠고, 점점 데이터를 분석하려는 노력을 하다 보니 pandas dataframe이 사용하기 매우 편리하다는 것을 알게 되었다.
   3. 그러나 텍스트 전처리는 달랐다. 사용할 줄 알아야 하는 모듈과 함수가 너무 많았으며, 특히나 word2vec의 과정이 너무나 힘들었기 떄문에 공부해본 내용을 정리해 보고자 한다.

      - 내가 텍스트 전처리를 이해함에 있어서 도움을 준 대부분의 소스는 [URl]'https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing/notebook' 이분 것을 참고를 했다.
       - 따라서 나도 같이 tweeter데이터셋을 이용해서 문자열 전처리를 연습해 보고(이는 csv파일을 다룰때에 지저분한 문자열이 column에 있을 경우 필수 기술이다) 나중에 셰익스피어의 희곡, 조선왕조 실록, 등 복잡한 데이터에 적용이 바로 가능하도록 익힐 것이다.
       
       
### 2. Encoding이란?
   - 제일 기본적인 의미는 text를 numertic type로 변환하는 것이다.
   - 이는 범주형 특성을 인코딩할 때에 매우 중요하게 작용한다.
   - 여기에는 대표적으로 one-hot-encoding과 embedding이 있는데
        1. one-hot-encoding은 예를 들면 ```['thank', 'you', 'hello']```가 있을때 단어들을 [[1,0,0],[0,1,0],[0,0,1]]이런 식의 벡터로 바꾸어 준다.
        2. 그러나 여기에는 *단어간의 유사도*를 고려해 주지 않는다는 문제가 존재한다.
        3. Embedding은 이를 해결할 수 있는 것으로, *유사도를 기준으로 단어의 벡터를 형성*하는 것이다.
   #### 2-1. One-Hot-Encoding 
      - 일반적으로 범주의 개수가 10개 이하일때 사용
   #### 2-2. Word2Vec
      - Embedding의 방법중 하나가 word2vec이다.
      - word2vec deeplearning model은 text자체가 onehotenncoding된 단어 벡터를 input과 output에 이용하게 된다.
         - one-hot-encoding 벡터의 크기는 우리가 '변경하고자 하는 단어의 개수'와 동일해야 한다.
         - embedding demension의 크기는 2로 정하면 나중에 2차원 평면에 관계도를 표현하기 쉽다.
   #### 2-3. Keras Layers
      - `keras.layers.experimental.preprocessing.Normalization`
      - `keras.layers.Standardization`
      - `keras.layers.Discretization`
      - `keras.layers.PreprocessingLayer`
      - `keras.layers.experimental.preprocessing.TextVectorization`
### 3. Text Preprocessing
  - [소스 코드]https://github.com/penguin1109/Machine-Learning-via-Tensorflow2.0/blob/master/RNN%20TensorFlow%20Lab/Text_Preprocessing(Twitter_Dataset).ipynb

