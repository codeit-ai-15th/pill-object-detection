# Dataset Description

## Datasets
- 원본 데이터 출처: [Ai Hub 경구약제 이미지 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=576) (원본 데이터를 가공 후 제공하였습니다.)
- Ai Hub에 있는 데이터를 추가로 사용 가능하나, 다만 데이터셋별로 포함된 클래스 정보가 다를 수 있으니 확인하고 모델링을 진행해 주세요.
- 단, 아래 AI hub 내 두가지 데이터 셋은 사용을 금지합니다.
    - Training > 라벨링데이터 > 경구약제조합 5000종 > TL_2_조합.zip
    - Training > 원천데이터 > 경구약제조합 5000종 > TS_2_조합.zip
- 이미지 는 PNG 형식, 주석은 COCO 포맷(JSON)으로 제공되며, 각 알약의 위치(Bonding Box)와 클래스 정보가 포함되어 있습니다.

## Files
- train_images - 훈련 데이터(이미지)
- train_annotations - 훈련 데이터(annotation)
- test_images - 테스트 데이터(이미지)

> Test Data에는 Train Data엔 없는 클래스가 존재하며, 제공된 데이터에는 크고 작은 오류들이 포함되어있습니다. 적합한 전처리와 데이터 보완을 통해 고득점을 노려보시길 바랍니다!

## Annotations (JSON 파일 구조)
> 필수 여부: M(Mandatory), O(Optional), C(Conditional)

| 구분 | 타입 | 필수여부 | 항목명 |
|---|---|---|---|
| images | Object | M | 약제 이미지정보 |
| images[].id | Number | M | 약제 이미지식별자 |
| images[].width | Number | M | 약제 이미지너비 |
| images[].height | Number | M | 약제 이미지높이 |
| images[].file_name | String | M | 약제 이미지파일명 |
| images[].drug_N | String | M | 알약코드 |
| images[].drug_S | String | M | 알약상태 |
| images[].back_color | String | M | 촬영배경 |
| images[].drug_dir | String | M | 알약방향 |
| images[].light_color | String | M | 촬영조명 |
| images[].camera_la | Number | M | 카메라위도 |
| images[].camera_lo | Number | M | 카메라경도 |
| images[].size | Number | M | 이미지 사이즈 |
| images[].dl_idx | String | M | 알약 식별자 |
| images[].dl_mapping_code | String | M | 제품코드 |
| images[].dl_name | String | M | 제품명 |
| images[].dl_name_en | String | O | 제품명(영어) |
| images[].img_key | String | M | 이미지 링크 |
| images[].dl_material | String | M | 성분명 |
| images[].dl_material_en | String | O | 성분명(영어) |
| images[].dl_custom_shape | String | M | 제조 모양 |
| images[].dl_company | String | M | 제조사명 |
| images[].dl_company_en | String | O | 제조사명(영어) |
| images[].di_company_mf | String | M | 위탁제조사명 |
| images[].di_company_mf_en | String | O | 위탁제조사명(영어) |
| images[].item_seq | Number | M | 품목기준코드 |
| images[].di_item_permit_date | Date | O | 허가일자 |
| images[].di_class_no | String | M | 약품 분류 |
| images[].di_etc_otc_code | String | M | 전문의약품/일반의약품 |
| images[].di_edi_code | String | M | EDI 코드 |
| images[].chart | String | M | 알약 제형 |
| images[].drug_shape | String | M | 알약 모양 |
| images[].thick | Number | M | 알약 두께 |
| images[].leng_long | Number | M | 알약 장축 |
| images[].leng_short | Number | M | 알약 단축 |
| images[].print_front | String | C | 식별문자_앞 |
| images[].print_back | String | C | 식별문자_뒤 |
| images[].color_class1 | String | M | 색상 1 |
| images[].color_class2 | String | O | 색상 2 |
| images[].line_front | String | M | 알약 앞면분할선 여부 |
| images[].line_back | String | M | 알약 뒷면분할선 여부 |
| images[].img_regist_ts | Date | O | 약학정보원 이미지 생성일 |
| images[].form_code_name | String | M | 정제 분류명 |
| images[].mark_code_front_anal | String | M | 알약 앞면마크 형태 |
| images[].mark_code_back_anal | String | M | 알약 뒷면마크 형태 |
| images[].mark_code_front_img | String | M | 알약 앞면마크 이미지 |
| images[].mark_code_back_img | String | M | 알약 뒷면마크 이미지 |
| images[].mark_code_front | String | M | 알약 앞면마크 코드 |
| images[].mark_code_back | String | M | 알약 뒷면마크 코드 |
| images[].change_date | Date | O | 변경일자 |
| type | type | O | json 타입 |
| annotations | Object | M | 라벨링정보 |
| annotations[].area | Number | M | 바운딩박스 면적 |
| annotations[].iscrowd | Number | O | 평가 분류 |
| annotations[].bbox | List | M | bbox 좌표 |
| annotations[].category_id | Number | M | category 아이디 |
| annotations[].ignore | Number | O | 무시 여부 |
| annotations[].segmentation | List | O | 라벨링바운딩 박스 |
| annotations[].image_id | Number | M | 이미지 아이디 |
| annotations[].id | Number | M | 어노테이션 아이디 |
| categories | Object | M | 라이선스 |
| categories[].supercategory | String | M | 슈퍼 카테고리 |
| categories[].id | Number | M | 카테고리 아이디 |
| categories[].name | String | C | 카테고리 명 |

## 데이터 가공 관련 참고사항
- 원본 데이터에서는 모든 image_id, category_id, annotation_id 값이 1로 일괄 설정되어 있었으나, 이번에 제공되는 데이터는 보다 정확한 관리를 위해 아래와 같이 수정되었습니다.
    - image_id: 각 이미지의 파일명(file_name)을 기준으로 중복 없이 고유한 ID가 순차적으로 부여됩니다.
    - category_id: 각 JSON 파일 내 첫 번째 이미지에 포함된 dl_idx 값을 사용하여 정수형으로 할당됩니다.
    - annotation_id: 각 어노테이션에 대해 순차적으로 증가하는 고유 ID가 부여됩니다. 단, 어노테이션이 유효한 경우에만 적용됩니다. 여기서 "존재"와 "유효"의 기준은 다음과 같습니다:
        - 존재: 어노테이션 항목에 "bbox"라는 키가 실제로 있어야 하며, 값이 비어 있지 않아야 합니다. "bbox" 키가 없거나 비어 있으면 해당 어노테이션은 제외됩니다.
        - 유효: "bbox"의 값은 리스트 형태이며, 리스트의 길이가 정확히 4여야 합니다. 이는 [x, y, width, height] 형식을 의미하며, 이 조건을 만족해야 어노테이션에 ID가 부여됩니다.