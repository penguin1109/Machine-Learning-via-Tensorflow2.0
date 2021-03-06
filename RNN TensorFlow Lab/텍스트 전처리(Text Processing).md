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
        3. Embedding은 이를 해결할 수 있는 것으로, *가중치를 기준으로 단어의 벡터를 형성*하는 것이다.
         
        1. Embedding Layers
            - 자연어 처리를 하려고 할 때 갖고 있는 훈련데이터의 단어들을 임베딩 층을 구현하여 임베딩 벡터로 학습하는 경우가 존재한다.
            - 이떄 keras의 Embedding()이라는 도구는 '사전훈련된 워드 임베딩'이기 떄문에 사용하기 좋다.
            - 임베딩 층은 일종의 lookup table의 역할을 하는데, 이는 입력정수(입력 시퀀스의 각 단어는 무조건 정수 인코딩이 되어 있어야 한다.)에 대해 밀집 벡터로 mapping하소 이 밀집 벡터는 인공 신경망의 학습 과정에서 가중치가 학습되는 것과 같은 방식으로 훈련이 된다.
               - 훈련 과정에서 단어는 모델이 해결하고자하는 작업에 맞는 값으로 업데이트가 되며, 이렇게 바뀐 밀집 벡터를 임베딩 벡터라고 한다.
               - lookup table은 단어 집합의 크기만큼의 행을 가지기 때문에 모든 단어는 고유한 임베딩 층을 갖는다.
               - 특정 단어가 정수 인코딩이 된 후에 테이블로부터 해당 인덱스에 위치한 임베딩 벡터를 꺼내 올 수 있께되는데, 이 임베딩 벡터는 모델의 입력이 되고, 역전파 과정에서 이 단어의 임베딩 벡터 값이 가중치를 고려하여 학습이 되는 것이다.
                  - 주의할 것은 무조건 단어들을 원핫인코딩을 거쳐서 입력값으로 반환할 필요가 없다는 것이다.
                  - 임베딩 벡터의 차원은 우리가 직접 선정할 수 있으며, 임베딩 층이 받는 parameter는 ```(vocab_size, output_dim, input_length)```가 있는데 이는   
                  ```(텍스트 데이터의 전체 단어 집합의 크기, 워드 임베딩 후 임베딩 벡터의 차원, 입력 시퀀스의 길이 = 입력 문장의 길이)```이다.
            
        
               
### 3. Text Preprocessing
  - [소스 코드]https://github.com/penguin1109/Machine-Learning-via-Tensorflow2.0/blob/master/RNN%20TensorFlow%20Lab/Text_Preprocessing(Twitter_Dataset).ipynb

