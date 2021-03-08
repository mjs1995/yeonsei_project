# 2020_데이터청년캠퍼스_연세대학교_프로젝트
# intro
- Covid-19로 인한 사회, 경제적 불황 때문에 대도시 중심으로 인구이동현상이 증가함에따라 중소도시의 인구 유출이 심화되고 있으며, 지방소멸에 대한 논의가 대두되고 있음

# objective
- 지방소멸 위기를 인식하고 현 상태를 진단 및 지방소멸 위험지역을 분류하고 동시에 지방소멸에 미치는 영향 요인들을 구체화.

# 역할
- DB 설계

        * MariaDB와 HeidiSQL을 이용해서 데이터 베이스 구축
        * MySQL WorkBench를 이용하여 ERD도식화

- 데이터 eda & FE

        * shap파일 전처리를 통해 geojson파일을 가지고 mapbox, folium, plotly를 통한 지도시각화
        * Cartogram을 이용한 라벨링된 지도 시각화
        * 피처 시각화
        * VIF 측정

- 데이터 모델링 
        
        * 교차 검증(Cross validation) - StratifiedKFold
        * GradientBoosting 
            -  RandomizedSearchCV이용한 파라미터 튜닝 및 f1_score, confusion_matrix 시각화
            -  random_state를 변경해가며 voting
        *  RandomForest
            -  RandomizedSearchCV이용한 파라미터 튜닝 및 f1_score, confusion_matrix 시각화
            -  random_state를 변경해가며 voting
        *  Xgboost
            -  feature importance, Permutation Importance , graphviz 시각화
            -  RandomizedSearchCV이용한 파라미터 튜닝 및 f1_score, confusion_matrix 시각화
            -  random_state를 변경해가며 voting 
        * Xgboost와 GradientBoosting 앙상블 

## 1) Data from Kosis
- 2015~2018 전국 228개 지방자치단체 데이터
Cofog(정부기능분류)기반으로
1. 주거 및 지역사회건설(하수도보급률, 상수도보급률, 주택수, 교통문화지수, 운전행태영역, 교통안전영역, 보행행태영역, 등)
2. 일반 공공행정(혼인건수, 조혼인율, 합계출산율, 평균연령, 남녀성비, 인구증가율, 당해 년 총 인구, 전년 총 인구, 등)
3. 보건(병원 수, 보건소 수, 상급종합병원 수, 약국 수, 요양병원 수, 의원 수, 종합병원 수, 치과병원 수, 한방병원 수, 등)
4. 사회보호(고위험음주율, 비만율, 건강상태 표준화지수, 주관적건강수준인지율, 자살률, 등)
5. 경제활동(도소매업사업체수, 도소매업종사자수, 서비스업사업체수, 서비스업종사자수, 등) 
6. 교육(교원1인당 학생수, 재적 학생수, 유치원 교원수, 유치원 수 , 사설학원수, 등)
7. 환경보호(녹지지역면적, 등)
8. 휴양, 문화, 종교(문화기반시설 수, 등)
9, 공공질서 및 안전(교통사고, 화재, 범죄, 자연재해, 생활안전, 자살, 감염병, 등)로 분류하여 자료를 구성 

## 2) News-article data
- 중앙일보, 경향신문에서 '지방자치'라는 키워드로 검색된 최근 1만개의 뉴스기사 수집
- 지방자치단체 명을 키워드로 검색한 최근 1년간의 지차체별 뉴스기사 수집(지자체별로 갯수 상이)

## 3) dataset
- data15~19 : 연도별 지방자치단체 데이터
- train :  15,16,17년도의 train set
- test : 18년도 test set 
- 모델링_train : 전처리 후 모델 학습에 사용될 train set
- 모델링_test : 전처리 후 모델에 입력될 최종 test set
- 소멸위험시군구_최종 : test셋의 결과
- 감성분석 데이터 : ''지방자치'라는 키워드로 검색된 20000개의 뉴스기사 데이터
- [신문사] 지방 데이터 : "지자체 명'을 키워드로 검색된 최근 1년간의 뉴스기사 데이터
- 고성감성분석, 곡성감성분석, 서천감성분석, 평창감성분석 : 소멸위험지역 3곳과 위험도가 낮은 1곳(평창군)의 뉴스기사 감성분석 결과

# Enviornment & require
- Python 3.7
## library
### machine learning
- pandas:  1.1.2
- matplotlib:  3.3.2
- seaborn:  0.11.0
- folium:  0.11.0
- plotly:  4.10.0
- sklearn:  0.23.2
- patsy:  0.5.1
- statsmodels:  0.12.0
- xgboost:  1.2.0
- lightgbm:  3.0.0
- eli5:  0.10.1

### BERT, Sentiment analysis
wget : '3.2'
tensorflow : '2.0.0'
tensorflow_hub : '0.9.0'

### word cloud
konlpy : '0.5.2'
re : '2.2.1'
wordcloud : '1.8.0'

### crawling 
selenium : '3.141.0'
tqdm : '4.49.0'

# how to use
## 1) machine learning model
### 1. eda 연도별 피처 비교
    def year_plot(e):
      plt.subplots(figsize=(20, 10))
      bar_width = 0.35
      a = dt_15.groupby(["sido"])[e].mean().sort_values(ascending=False)
      a.plot.bar(rot=0, figsize=(20, 10), label='2015', color='yellowgreen')
      b = dt_16.groupby(["sido"])[e].mean().sort_values(ascending=False)
      b.plot.bar(rot=0, figsize=(20, 10), label='2016', color='y')
      c = dt_17.groupby(["sido"])[e].mean().sort_values(ascending=False)
      c.plot.bar(rot=0, figsize=(20, 10), label='2017', color='azure')
      d = test.groupby(["sido"])[e].mean().sort_values(ascending=False)
      d.plot.bar(rot=0, figsize=(20, 10), label='2018', color='forestgreen')
      plt.legend()
      plt.title('연도별 '+str(e))
      return plt.show()

### 2. 모델의 f1_score 비교 함수 
    def four_f1(model_f1,model):
      print("F1 Cross_validate",model_f1)
      print("F1 Macro:",f1_score(test_target, model, average='macro'))
      print("F1 Micro:",f1_score(test_target, model, average='micro'))  
      print("F1 Weighted:",f1_score(test_target, model, average='weighted'))
      print("\nMatrix of confusion")
      return confusion_matrix(test_target, model)

### 3. 모델의 confusion_matrix 시각화 함수
    def cnf_matrix_model(model):
      cnf_matrix_gbc = confusion_matrix(test_target, model)
      g = sns.heatmap(pd.DataFrame(cnf_matrix_gbc), annot=True, cmap="BuGn", fmt='g')
      buttom , top = g.get_ylim()
      g.set_ylim(buttom+0.5, top-0.5)
      plt.ylabel('Actual Label')
      plt.xlabel('Predicted Label')
      return g


### 4. 변수 중요도     
    def plot_feature_importance(model, X_train, figsize=(12, 6)):
    sns.set_style('darkgrid')
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.figure(figsize=figsize)
    plt.rc("font",family="Malgun Gothic")
    plt.rc("axes",unicode_minus=False)
    
    plt.barh(pos, feature_importance[sorted_idx], align='center',color = 'teal' )
    plt.yticks(pos, X_train.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


### 5. 앙상블 코드 
    def combine_voters(data, weights=[0.5, 0.5]):
      vc.voting="soft"
      vc1_probs = vc.predict_proba(data)
      vc2.voting="soft"
      vc2_probs = vc2.predict_proba(data) 
    
      final_vote = (vc1_probs * weights[0]) + (vc2_probs * weights[1])
      predictions = np.argmax(final_vote, axis=1)
    
      return predictions

## 2) news-articles analysis

** BERT 모델은 google colab을 이용하여 구현하였습니다. colab으로 실행시켜주시는것을 권장해드립니다.**

** jupyter notebook 실행시) 
#!pip install wget
import wget
url = "https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py"
wget.download(url)

위와 같은 방법으로 실행시켜주시기 바랍니다.


### help Function
#### 1. make bert_encode : word piece tokenizer. BERT의 incoder 부분을 구현
    def bert_encode(texts, tokenizer, max_len=128):
      all_tokens = []
      all_masks = []
      all_segments = []
    
      for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
      return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

#### 2. make build_model : BERT모델의 fine tuning architecture 구현
    def build_model(bert_layer, max_len=128):
      input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
      input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
      segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

      _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
      clf_output = sequence_output[:, 0, :]
      out = Dense(1, activation='sigmoid')(clf_output)
    
      model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
      model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
      
      return model

![제목 없음](https://user-images.githubusercontent.com/47103479/92443807-f993a480-f1ec-11ea-8960-acdde191309c.png)
![제목 없음1](https://user-images.githubusercontent.com/47103479/92443823-01534900-f1ed-11ea-811c-9ebf9165443b.png)
![제목 없음2](https://user-images.githubusercontent.com/47103479/92443845-09ab8400-f1ed-11ea-92d8-c02ff3e8d8b0.png)
![제목 없음3](https://user-images.githubusercontent.com/47103479/92443881-18923680-f1ed-11ea-81a5-de87f90ec71a.png)
![제목 없음4](https://user-images.githubusercontent.com/47103479/92443897-1c25bd80-f1ed-11ea-8fee-2fe60bd5e87b.png)
![제목 없음5](https://user-images.githubusercontent.com/47103479/92443904-1fb94480-f1ed-11ea-9d51-6f30fb13c051.png)
![제목 없음6](https://user-images.githubusercontent.com/47103479/92443908-21830800-f1ed-11ea-8d54-2dab04bf8b76.png)
![제목 없음7](https://user-images.githubusercontent.com/47103479/92443912-22b43500-f1ed-11ea-8995-4e4918afbc7b.png)
![제목 없음8](https://user-images.githubusercontent.com/47103479/92443918-247df880-f1ed-11ea-97bd-6d29dbf04a63.png)
![제목 없음9](https://user-images.githubusercontent.com/47103479/92443921-25af2580-f1ed-11ea-883f-00756d04b170.png)
![제목 없음10](https://user-images.githubusercontent.com/47103479/92443931-2942ac80-f1ed-11ea-87fa-680e2d3fd3fd.png)
![제목 없음11](https://user-images.githubusercontent.com/47103479/92443938-2a73d980-f1ed-11ea-9427-b5ba3b2b8f0f.png)
![제목 없음12](https://user-images.githubusercontent.com/47103479/92443942-2c3d9d00-f1ed-11ea-9ca4-8c57d7d67940.png)
![제목 없음13](https://user-images.githubusercontent.com/47103479/92443947-2d6eca00-f1ed-11ea-8f62-71f358cfec47.png)
![제목 없음14](https://user-images.githubusercontent.com/47103479/92443949-2e9ff700-f1ed-11ea-84c3-349187508141.png)
![제목 없음15](https://user-images.githubusercontent.com/47103479/92443959-319ae780-f1ed-11ea-88df-520a68c48e90.png)
![제목 없음16](https://user-images.githubusercontent.com/47103479/92443965-3364ab00-f1ed-11ea-880d-475630226af3.png)

