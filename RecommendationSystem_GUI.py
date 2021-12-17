import streamlit as st
import pandas as pd
# import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.2f}'.format

## data
link_source = 'https://drive.google.com/uc?export=download&id='

link_product = link_source + '1QGEVPuV34xIfZMadexbnu3u1o4L_heYz'
link_review = link_source + '1Qd2j-SP0IZN_MOJ4lDbhN7dthDWKxdTS'
link_data_xl = link_source + '1Q5xnXFPHDENDfhjLx6RH1AYCkcz1y90b'
link_Recomender_Collborative = link_source + '1QSuaLQ8OInj3LAHl3aHMvNqrLwjZYHNL'



#--------------
# Gonfig GUI
st.set_page_config(page_title='Product Recommendation', layout = 'wide', initial_sidebar_state = "expanded",page_icon="📝")

# Load data
@st.cache
def load_products():
    return pd.read_csv(link_product)

products = load_products() 

@st.cache
def load_reviews():
    return pd.read_csv(link_review)

@st.cache
def load_data_xl():
    return pd.read_csv(link_data_xl, encoding='UTF-8')

@st.cache
def load_cosine_similarities():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
    tf = TfidfVectorizer(analyzer='word', min_df=0)
    tfidf_matrix = tf.fit_transform(data_xl.products_wt)
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

@st.cache
def load_recomenders_collborative():
    return pd.read_csv(link_Recomender_Collborative)

# Function recommenders for an user
@st.cache
def get_recommenders_for_user(recommenders,customer_id, items_number=6):
    recds = recommenders[recommenders['customer_id']==customer_id]
    recds = recds.head(items_number)
    result = pd.merge(left=recds,right=products[products.item_id.isin(recds.product_id)] ,how='left',left_on='product_id',right_on='item_id')
    return result

# Set option show chart
st.set_option('deprecation.showPyplotGlobalUse', False)

#--------------
# GUI
menu = ["Tổng quan","Tìm hiểu dữ liệu","Đề xuất người dùng với Content based filtering","Đề xuất người dùng với Collaborative filtering"]
choice = st.sidebar.selectbox("Chọn chức năng:",menu)
if choice == 'Tổng quan':
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h1 style="color:white;text-align:center;">ĐỒ ÁN TỐT NGHIỆP DATA SCIENCE</h1>
    <h1 style="color:white;text-align:center;">HỆ THỐNG ĐỀ XUẤT SẢN PHẨM CHO NGƯỜI DÙNG</h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; color: #003153'>TỔNG QUAN HỆ THỐNG ĐỀ XUẤT NGƯỜI DÙNG</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: #0047ab;'>1. Giới thiệu</h3>", unsafe_allow_html=True)
    st.markdown("- Tiki là một hệ sinh thái thương mại “all in one”, trong đó có **tiki.vn**, là một website thương mại điện tự đứng top 2 của Vietnam và top 6 khu vực Đông Nam Á")   
    st.markdown("- Trên trang này đã triển khai nhiều tiện ích hỗ trợ nâng cao trải nghiệm người dùng và họ muốn xây dựng nhiều tiện ích hơn nữa.")
    st.markdown("- Giả sử công ty này chưa triển khai Recommender System và bạn được yêu cầu triển khai hệ thống này, bạn sẽ làm gì?")
    st.write(" ")

    st.markdown("<h3 style='text-align: left; color: #0047ab;'>2. Vì sao có dự án?</h3>", unsafe_allow_html=True)
    st.markdown("- Chưa có hệ thống Recommendation System")   
    st.markdown("- => Mục tiêu:Xây dựng Recommendation System cho một hoặc một số nhóm hàng hóa trên tiki.vn giúp đề xuất và gợi ý cho người dùng/ khách hàng. => Xây dựng các mô hình đề xuất")   
    st.write(" ")

    st.markdown("<h3 style='text-align: left; color: #0047ab;'>3. Dữ liệu cung cấp</h3>", unsafe_allow_html=True)
    st.markdown("- Dữ liệu được cung cấp sẵn gồm có các tập tin: ProductRaw.csv, ReviewRaw.csv chứa thông tin sản phẩm, review và rating cho các sản phẩm thuộc các nhóm hàng hóa như Mobile_Tablet, TV_Audio, Laptop, Camera, Accessory")   
    st.image('picture/tables properties.jpg',caption='Mô tả các thuộc tính')
    st.write(" ")
    
    st.markdown("<h3 style='text-align: left; color: #0047ab;'>4. Yêu cầu đặt ra là tập trung giải quyết hai bài toán:</h3>", unsafe_allow_html=True)
    st.markdown("- Bài toán 1: Đề xuất người dùng với content - based filtering")
    st.markdown("- Bài toán 2: Đề xuất người dùng với Collaborative filtering")
    st.image('picture/two recomender system.png')
    st.write(" ")

elif choice == "Tìm hiểu dữ liệu":
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h1 style="color:white;text-align:center;">HỆ THỐNG ĐỀ XUẤT SẢN PHẨM CHO NGƯỜI DÙNG</h1>
    <h2 style="color:white;text-align:center;">Tìm hiểu dữ liệu</h2>
    </div>
    """
    strNhanXet ='<h5 style="color:#0047ab;text-align:left;">Nhận xét:</h5>'
    # Header-----
    st.markdown(html_temp,unsafe_allow_html=True)
    reviews = load_reviews() # pd.read_csv('data/Review.csv')
    
    sns.displot(products,x='rating',kind='hist')
    plt.title("Biểu đồ phân phối rating của product",fontsize=8,color='blue')
    st.pyplot()
        
    st.markdown(strNhanXet,unsafe_allow_html=True)
    st.write('''- Rating của sản phẩm trong product có giá trị từ 0 đến 5
- Số lượng rating 0 và 5 tương đương nhau
- Điểm rating phần lớn tập trung từ 4-5.
- Điểm rating = 0 có thể do sản phẩm chưa được đánh giá nên để mặc định''')
    st.write('')

    # Sự phân bổ Ratings của khách hàng
    sns.displot(reviews,x='rating',kind='hist')
    plt.title("Biểu đồ phân phối rating của khách hàng",fontsize=8,color='blue')
    st.pyplot()
    st.markdown(strNhanXet,unsafe_allow_html=True)
    st.write('''- Phần lớn khách hàng phản hồi tích cực về sản phẩm
- Sản phẩm có chất lượng tốt hoặc khách hàng dễ tính
- Đa phần các đánh giá từ trên 4 điểm, đánh giá 5 điểm chiếm tỷ lệ cao
- Các đánh giá có giá trị từ 1-5''')
    st.write('')

    # Sản phẩm theo thương hiệu
    brands = products.groupby('brand')['item_id'].count().sort_values(ascending=False)
    bar = sns.barplot(data=brands.to_frame()[1:11].reset_index(),x='brand',y='item_id', palette="Blues_r")
    bar.set_xticklabels(bar.get_xticklabels(), rotation=90)
    plt.ylabel('Number of product')
    plt.title("Top 10 thương hiệu có nhiều sản phẩm nhất",fontsize=8,color='blue')
    st.pyplot()
    st.markdown(strNhanXet,unsafe_allow_html=True)
    st.write('''- Thương hiệu samsung có nhiều mã hàng nhất
- Các thương hiệu khác có số lượng mã hàng khoảng từ 60-100''')
    st.write("")

    # Giá bán theo thương hiệu
    price_by_brand = products.groupby(by='brand').mean()['price']
    plt.figure(figsize=(15,8))
    bar = sns.barplot(data=price_by_brand.sort_values(ascending=False)[:10].to_frame().reset_index(),x='brand',y='price', palette="Blues_r")
    bar.set_xticklabels(bar.get_xticklabels(), rotation=90)
    plt.ylabel('Price')
    plt.title("Top 10 giá bán theo thương hiệu",fontsize=18,color='blue')
    st.pyplot()
    st.markdown(strNhanXet,unsafe_allow_html=True)
    st.write('- Thương hiệu Hitachi có trung bình giá bán cao nhất')
    st.write("")
    
    top_rating_products = reviews.groupby(by='product_id').count()['customer_id'].sort_values(ascending=False)[:20]
    top_rating_products.index =products[products.item_id.isin(top_rating_products.index)]['name'].str[:25]
    plt.figure(figsize=(15,8))
    bar = sns.barplot(data=top_rating_products.to_frame().reset_index(),x='name',y='customer_id', palette="Blues_r")
    bar.set_xticklabels(bar.get_xticklabels(), rotation=90)
    plt.ylabel('Số lượng đánh giá')
    plt.title("Top 20 sản phẩm được đánh giá nhiều nhất",fontsize=18,color='blue')
    st.pyplot()
    st.markdown(strNhanXet,unsafe_allow_html=True)
    st.write('''- Phụ kiện điện thoại, máy tính
- Chuột không dây logitech được đánh giá nhiều nhất''')

elif choice == "Đề xuất người dùng với Content based filtering":
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h1 style="color:white;text-align:center;">HỆ THỐNG ĐỀ XUẤT SẢN PHẨM CHO NGƯỜI DÙNG</h1>
    <h2 style="color:white;text-align:center;">Đề xuất sản phẩm cho người dùng với Content based filtering</h2>
    </div>
    """
    # Header-----
    st.markdown(html_temp,unsafe_allow_html=True)
    data_xl = load_data_xl()
    cosine_similarities = load_cosine_similarities()
    with st.form(key='Tìm kiếm sản phẩm tương tự'):
        st.write("###### Lựa chọn tên sản phẩm")
        name_item = st.selectbox('Chọn tên sản phầm cần tìm các sản phẩm tương tự', products['name'])
        items_num = st.slider(label='Số lượng sản phẩm đề xuất:',min_value=2,max_value=8,value=8,step=2)
        submit_button = st.form_submit_button(label='Đề xuất sản phẩm 📝')
    if submit_button:
        results = {}
        for idx, row in data_xl.iterrows():
            similar_indices = cosine_similarities[idx].argsort()[-9:-1]
            similar_items = [(cosine_similarities[idx][i]) for i in similar_indices]
            similar_items = [(cosine_similarities[idx][i], data_xl.index[i]) for i in similar_indices]
            results[idx] = similar_items[0:]
        ## lựa chọn sản phẩm
        st.markdown("<h3 style='text-align: left; color: #0047ab;'>Thông tin sản phẩm được chọn:</h3>", unsafe_allow_html=True)
        idx_prd = products.index[products['name']==name_item].tolist()[0]
        ## đưa hình vào form
        col1, col2 = st.columns([2,8])
        with col1:
            st.image(str(products.loc[idx_prd,'image']))
        with col2:
            st.write('###### %s'%(products.loc[idx_prd,'name']))
            strprice = '%s đ'%(format(products.loc[idx_prd,'price'],',d'))
            st.markdown("<h4 style='text-align: left; color: Red;'>"+strprice+"</h4>", unsafe_allow_html=True)
            st.write('Thương Hiệu: %s'%(products.loc[idx_prd,'brand']))
            st.write('Đánh gía: %s'%(products.loc[idx_prd,'rating']))
            st.markdown("[Web link](https://tiki.vn/"+products.loc[idx_prd,'url'].split('//')[-1]+")")
        st.write("")

        # Kết quả sản phẩm tương tự
        st.markdown("<h3 style='text-align: left; color: #0047ab;'>Có thể bạn muốn xem các sản phẩm này:</h3>", unsafe_allow_html=True)
        sim_list = []
        for i in range(0,items_num,2):
            # for i
            item_id = data_xl.iloc[results[idx_prd][i][1]]['item_id']
            idx_prd = products.index[products['item_id']==item_id].tolist()[0]
            # for i+1
            item_id_1 = data_xl.iloc[results[idx_prd][i+1][1]]['item_id']
            idx_prd_1 = products.index[products['item_id']==item_id_1].tolist()[0]
            # in sản phẩm          
            col1, col2, col3, col4 = st.columns([2,3,2,3])
            with col1:
                st.image(str(products.loc[idx_prd,'image']))
            with col2:
                st.write('###### %s'%(products.loc[idx_prd,'name']))
                strprice = '%s đ'%(format(products.loc[idx_prd,'price'],',d'))
                st.markdown("<h4 style='text-align: left; color: Red;'>"+strprice+"</h4>", unsafe_allow_html=True)
                st.write('Thương Hiệu: %s'%(products.loc[idx_prd,'brand']))
                st.write('Đánh gía: %s'%(products.loc[idx_prd,'rating']))
                st.markdown("[Web link](https://tiki.vn/"+products.loc[idx_prd,'url'].split('//')[-1]+")")
            with col3:
                st.image(str(products.loc[idx_prd_1,'image']))
            with col4:
                st.write('###### %s'%(products.loc[idx_prd_1,'name']))
                strprice = '%s đ'%(format(products.loc[idx_prd_1,'price'],',d'))
                st.markdown("<h4 style='text-align: left; color: Red;'>"+strprice+"</h4>", unsafe_allow_html=True)
                st.write('Thương Hiệu: %s'%(products.loc[idx_prd_1,'brand']))
                st.write('Đánh gía: %s'%(products.loc[idx_prd_1,'rating']))
                st.markdown("[Web link](https://tiki.vn/"+products.loc[idx_prd_1,'url'].split('//')[-1]+")")
            st.write(" ")
    

elif choice == "Đề xuất người dùng với Collaborative filtering":
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h1 style="color:white;text-align:center;">HỆ THỐNG ĐỀ XUẤT SẢN PHẨM CHO NGƯỜI DÙNG</h1>
    <h2 style="color:white;text-align:center;">Đề xuất sản phẩm cho người dùng với Collaborative filtering</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    custIdsDefault = [10701688,11763074,13132598,11360428,12501987,12286743,15356393,14329244,14634037,16136723,17046856,13562927,12533916,10669721,10794654,10067279,11044205,14390035,18132106,11282171,14284583,17335575,14090397,15677717,19198887,17680741,18860589,19781287,11517697,16822674,10107865,19014688,11472727,11671593,13067276,15534235,15415309,16837996,20188312,17734805,20758207,11163955,11043040,12985305,16409336,11676531,11858104,17611104,10034801,16799850,18256187,10279553,19677744,18535485,11174386,17539237,16319435,11345636,15868853,13175665,10600682,13890346,17334166,10371235,18285577,11225533,11520401,10523440,10568749,13800460,18810996,14555930,13086421,15083755,10522112,14703210,13981024,10906890,13333245,15462201,13631188,20441329,13085882,13554619,19148341,13642827,14058815,10321855,10509329,20545482,20775881,19190448,20108289,11333760,19276521,17882495,14425088,14877474,17515216,10309831,10870410,10526674,13459468,11174641,11232488,19371818,15191237,18915491,15508611,13700067,18421965,13612033,15766998,10416934,12179361,12250233,10503680,19452549,10542562,13872637,11612099,17681785,12452208,10245727,17872615,11365720,11059384,15856762,10182179,18440429,14742850,20387094,12486134,10488355,14451729,10625224,15861135,11213649]
    recommenders = load_recomenders_collborative()
    with st.form(key='Đề xuất sản phẩm cho người dùng'):
        selected_user = st.multiselect('Chọn người dùng', custIdsDefault ,[10701688])
        items_num = st.slider(label='Số lượng sản phẩm đề xuất:',min_value=1,max_value=10,value=6,step=1)
        submit_button = st.form_submit_button(label='Thực hiện đề xuất sản phẩm 📝')
    if submit_button:
        if len(selected_user) ==0:
            st.markdown("<h4 style='text-align: left; color: #0047ab;'>Bạn chưa chọn người dùng, vui lòng chọn người dùng có trong danh sách!</h4>", unsafe_allow_html=True)
        else:
            data = get_recommenders_for_user(recommenders,selected_user[0],items_num)
            data.reset_index()
            strtemp = 'Có {} sản phẩm được đề xuất:'.format(data.shape[0])
            st.markdown("<h3 style='text-align: left; color: #0047ab;'>"+strtemp+"</h3>", unsafe_allow_html=True)
            # in sản phẩm   
            for i in range(0,data.shape[0],2):
                col1, col2, col3, col4 = st.columns([2,3,2,3])
                with col1:
                    st.image(str(data.loc[i,'image']))
                with col2:
                    st.write('###### %s'%(data.loc[i,'name']))
                    strprice = '%s đ'%(format(data.loc[i,'price'],',d'))
                    st.markdown("<h4 style='text-align: left; color: Red;'>"+strprice+"</h4>", unsafe_allow_html=True)
                    st.write('Thương Hiệu: %s'%(data.loc[i,'brand']))
                    st.write('Đánh gía: %s'%(data.loc[i,'rating']))
                    st.markdown("[Web link](https://tiki.vn/"+data.loc[i,'url'].split('//')[-1]+")")
                with col3:
                    if (i+1) < data.shape[0]:
                        st.image(str(data.loc[i+1,'image']))
                with col4:
                    if (i+1) < data.shape[0]:
                        st.write('###### %s'%(data.loc[i+1,'name']))
                        strprice = '%s đ'%(format(data.loc[i+1,'price'],',d'))
                        st.markdown("<h4 style='text-align: left; color: Red;'>"+strprice+"</h4>", unsafe_allow_html=True)
                        st.write('Thương Hiệu: %s'%(data.loc[i+1,'brand']))
                        st.write('Đánh gía: %s'%(data.loc[i+1,'rating']))
                        st.markdown("[Web link](https://tiki.vn/"+data.loc[i+1,'url'].split('//')[-1]+")")
                st.write("")
            # in bảng kết quả đề xuất 
            st.markdown("<h3 style='text-align: left; color: #0047ab;'>Bảng kết quả chi tiết:</h3>", unsafe_allow_html=True)
            st.dataframe(data)

# Footer
footer_name ='''<a style="color:#282c35; font-style:italic; font-size:14px; padding:10px 0 15px">
Trung Tâm Tin Học Trường Đại học khoa học Tự nhiên (Team: VÕ HỮU LỘC - PHẠM THANH TÙNG)</a>'''
st.markdown("<div><a style='color:#e5e4e2'>"+"_"*178+"</a><br>"+footer_name+"</div>",unsafe_allow_html=True)