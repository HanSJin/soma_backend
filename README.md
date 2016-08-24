# soma_backend
소프트웨어 마에스트로 백엔드 과제 - 이미정

## 결과 링크 주소
(http://125.180.52.121:18887)
위 주소는 집 공유기 포트포워딩을 통해 실행하였습니다.

## 평가서버 name명
평가 서버에서 제가 사용한 이름은 lmjing이며 최고 순위의 name값은 lmjing6입니다.
real_name에 이미정을 사용하였으며 없이 호출한 경우도 있어 name값을 알려드립니다.
이름으로 호출을 하려 하였으나 순위가 뒤쳐져 호출하지 않았습니다.

## docker
(https://hub.docker.com/r/lmjing/soma_backend/)

이미지 다운 : docker pull lmjing/
soma_classifier에서 
```{.python}
import nltk
nltk.download()
```
실행 할 경우 d -> book -> q를 차례대로 입력해주시면 됩니다.

##성능 개선 방법

1. konlpy를 통한 형태소 분석
2. 상품에서 분류하는데 도움이 되지 않는 stop word 제거
```{.python}
from konlpy.tag import Twitter
from nltk.corpus import stopwords

stop_words = stopwords.words('english')+ [u'1',u'2',u'3',u'4',u'5',u'6',u'7',u'8',u'9']

def set_konlpy(text):
    words = Twitter().pos(text)
    check = ['Alpha','Number','Noun']
    nn = [e[0] for e in words if e[1] in check if e[0] not in stop_words]
    return ' '.join(nn)

n_list = []
for each in d_list:
    n_list.append(set_konlpy(each))
    print (set_konlpy(each))
```
3. trigram을 사용 feature더 다양하게 추가
```{.python}
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1, 3),token_pattern=r'(?u)\b\w+\b', min_df=1)
x_list = vectorizer.fit_transform(n_list)
```
4. tfidf를 사용해 단어들의 중요도, 빈도수 파악
```{.python}
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
z_list = tfidf.fit_transform(x_list)
```

##이 외의 시도들
- https://github.com/irony/caffe-docker-classifier 모델을 이용 : docker로 irony/caffe-docker-classifier 다운 받은 후 실행, 이미지의 결과 값 5가지를 feature에 추가하여 학습하고자 하였으나 시간이 너무 많이 소모되어 시간 부족으로 인해 실제 테스트해보지는 못함.
```{.python}
d_list = []
cate_list = []
i_list = []
for each in train_df.iterrows():
    cate = ";".join([each[1]['cate1'],each[1]['cate2'],each[1]['cate3']])
    cate_list.append(cate)
    d_list.append(each[1]['name'])
    i_list.append(get_image_feature(each[0]))
```
위와 같이 feature을 추가하려 했음
