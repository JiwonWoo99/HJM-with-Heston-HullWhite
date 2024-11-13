이 저장소는 논문 "SOFR Term Structure Dynamics: Discontinuous Short Rates and Stochastic Volatility Forward Rates"의 모델을 구현한 코드와 필요한 데이터를 담았다. 
본 연구에서는 불연속적인 단기 금리(piecewise constant between FOMC)와 Heston/Hull-White을 고려한 SOFR 선도금리 구조의 수식과 모델링을 구현했다.
모델에 사용된 데이터는 블룸버그에서 제공받은 정보이며, 해당 데이터는 압축된 Data.zip 파일에 포함되어 있습니다. 이 파일을 다운로드한 후 압축을 풀어 데이터를 사용할 수 있다.
모델 실행 후 결과는 results 폴더에 저장되며, 각 결과는 val_date를 이름으로 하는 하위 폴더에 저장된다.
