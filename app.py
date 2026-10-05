import gradio as gr
import joblib
import numpy as np
import pandas as pd

# 1. تحميل النموذج وقائمة الميزات عند إقلاع التطبيق
model = joblib.load('nyc_rf_model.pkl')
trained_features = joblib.load('model_features.pkl')

# 2. الدالة المسؤولة عن معالجة المدخلات والتنبؤ
def evaluate_property(borough, building_category, year_built, gross_sqft, land_sqft, total_units, sale_month, tax_class):
    # أ. هندسة الميزات التلقائية
    age = max(0, 2017 - year_built)
    
    # تحديد الحقبة المعمارية
    if year_built < 1940:
        era = 'Pre-War (Historic)'
    elif year_built <= 1980:
        era = 'Post-War (Mid-Century)'
    elif year_built <= 2000:
        era = 'Late 20th Century'
    else:
        era = 'Modern'
        
    safe_units = max(1, total_units)
    unit_size = gross_sqft / safe_units
    tax_class_sale = f'Tax_Class_{tax_class}'
    
    # ب. إنشاء متجه المدخلات ومطابقته بدقة مع الـ 57 ميزة
    input_dict = {col: 0 for col in trained_features}
    
    # تعبئة الميزات الرقمية
    input_dict['LAND SQUARE FEET'] = land_sqft
    input_dict['GROSS SQUARE FEET'] = gross_sqft
    input_dict['TOTAL UNITS'] = total_units
    input_dict['AGE'] = age
    input_dict['UNIT_SIZE'] = unit_size
    input_dict['SALE_MONTH'] = sale_month
    
    # تفعيل أعمدة One-Hot المناسبة برقم 1
    categories_to_activate = [
        f'BOROUGH_{borough}',
        f'BUILDING CLASS CATEGORY_{building_category}',
        f'BUILDING_ERA_{era}',
        f'TAX_CLASS_SALE_{tax_class_sale}'
    ]
    for col_name in categories_to_activate:
        if col_name in input_dict:
            input_dict[col_name] = 1
            
    input_df = pd.DataFrame([input_dict])
    
    # ج. حساب تنبؤات كل شجرة من الـ 100 شجرة لقياس الثقة
    tree_preds = [np.expm1(t.predict(input_df))[0] for t in model.estimators_]
    
    estimated_price = np.mean(tree_preds)
    lower_bound = np.percentile(tree_preds, 10)
    upper_bound = np.percentile(tree_preds, 90)
    
    # احتساب نسبة الثقة استناداً لتشتت أصوات الأشجار
    std_dev = np.std(tree_preds)
    confidence_ratio = max(0, 1 - (std_dev / estimated_price))
    confidence_score = round(confidence_ratio * 100, 1)
    
    # تنسيق النتائج للعرض
    price_output = f"${estimated_price:,.0f}"
    range_output = f"${lower_bound:,.0f}  إلى  ${upper_bound:,.0f}"
    conf_output = f"{confidence_score}%"
    
    return price_output, range_output, conf_output

# 3. إعداد عناصر الواجهة وقوائم الاختيار
borough_choices = ['Manhattan', 'Brooklyn', 'Queens', 'Staten Island', 'Bronx']
category_choices = [
    '01 ONE FAMILY DWELLINGS',
    '02 TWO FAMILY DWELLINGS',
    '10 COOPS - ELEVATOR APARTMENTS',
    '13 CONDOS - ELEVATOR APARTMENTS',
    '07 RENTALS - WALKUP APARTMENTS'
]

# 4. بناء وتصميم واجهة Gradio
demo = gr.Interface(
    fn=evaluate_property,
    inputs=[
        gr.Dropdown(choices=borough_choices, value='Brooklyn', label="المنطقة (Borough)"),
        gr.Dropdown(choices=category_choices, value='01 ONE FAMILY DWELLINGS', label="نوع العقار (Building Category)"),
        gr.Slider(minimum=1850, maximum=2017, value=1950, step=1, label="سنة البناء (Year Built)"),
        gr.Number(value=2000, label="مساحة البناء بالقدم المربع (Gross Sq Ft)"),
        gr.Number(value=2500, label="مساحة الأرض بالقدم المربع (Land Sq Ft)"),
        gr.Number(value=1, label="إجمالي عدد الوحدات (Total Units)"),
        gr.Slider(minimum=1, maximum=12, value=6, step=1, label="شهر البيع المتوقع (Sale Month)"),
        gr.Radio(choices=['1', '2', '4'], value='1', label="الفئة الضريبية (Tax Class)")
    ],
    outputs=[
        gr.Textbox(label="السعر التقديري المتوقع (Estimated Price)", text_align="center"),
        gr.Textbox(label="النطاق السعري المرجح بثقة 80% (Confidence Interval)", text_align="center"),
        gr.Label(label="درجة ثقة وتماسك النموذج (Confidence Score)")
    ],
    title="🏢 نظام تسعير عقارات نيويورك الذكي (NYC Property Valuation AI)",
    description="أدخل مواصفات العقار للحصول على تقييم فوري مدعوم بنطاق الثقة الإحصائي لنموذج الغابات العشوائية.",
    theme="soft"
)

# 5. تشغيل السيرفر
if __name__ == "__main__":
    demo.launch()