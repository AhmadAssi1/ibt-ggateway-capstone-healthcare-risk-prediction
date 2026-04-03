import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
 
st.title("hello , this is Roa'a ")

name = st.text_input("Enter Your Name :")

st.header("This is a Header")
st.subheader("This is a Subheader")

st.text("This is a simple text.")
st.markdown("### Markdown: _Italic_, **Bold**, `Code`")

df = pd.DataFrame({
    'Column 1': [1, 2, 3],
    'Column 2': [4, 5, 6]
})

st.write(df)

st.table(df)  # Static table
st.dataframe(df)  # Interactive table


code = '''def say_hello():
    print("Hello, Streamlit!")
'''

st.code(code, language='python')

if st.button('Click Me!'):
    st.write("You clicked the button!")

name = st.text_input("Enter your name")
st.write(f"Hello, {name}!")

age = st.slider("Select your age", 0, 100, 25)
st.write(f"You are {age} years old.")

option = st.selectbox("Choose a fruit", ['Apple', 'Banana', 'Orange'])
st.write(f"You selected: {option}")

agree = st.checkbox("I agree")
if agree:
    st.write("Thank you for agreeing!")

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [10, 20, 30])
st.pyplot(fig)

df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length")
st.plotly_chart(fig)

chart = alt.Chart(df).mark_bar().encode(x='sepal_width', y='sepal_length')
st.altair_chart(chart)


st.sidebar.title("Options")
option = st.sidebar.selectbox("Choose an option", ['Option 1', 'Option 2', 'Option 3'])
st.write(f"You selected: {option}")


if 'count' not in st.session_state:
    st.session_state.count = 0

increment = st.button('Increment Count')

if increment:
    st.session_state.count += 1

st.write(f"Count: {st.session_state.count}")


uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
