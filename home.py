import streamlit as st

st.set_page_config(page_title="🌿 Carbon Emission Optimizer", layout="wide")

# Sidebar with icons and emojis
st.sidebar.title("🧭 Navigation Bar")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "⚖️ Carbon Emission Calculator",
        "🗺️ Route Visualization",
        "💡 Emission Optimization Tips",
        "🛠️ Admin Panel"
    ]
)

def load_home():
    st.title("🌿 Carbon Footprint Optimization in Supply Chain Logistics")
    st.markdown(
        """
        This application helps you **calculate** and **visualize** carbon emissions for your delivery routes.
        Select an option from the sidebar to get started. 🚛🌍

        <p style="font-size:16px; color:#e96e51;">🔧 Developed as Edunet ICBP 2.0 Program</p>
        """,
        unsafe_allow_html=True
    )

    st.image(
        "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1350&q=80",
        caption="Optimizing Routes for a Greener Future 🌱",
        use_column_width=True,
    )
    st.markdown("---")

    st.subheader("Why Optimize Carbon Emissions? 🌎")
    st.write(
        """
        - **Reduce environmental impact** by lowering greenhouse gas emissions.
        - **Cut costs** through improved fuel efficiency and optimized routes.
        - **Meet regulatory requirements** and corporate sustainability goals.
        - **Enhance brand reputation** by demonstrating environmental responsibility.
        """
    )

    st.image(
        "https://images.unsplash.com/photo-1518717758536-85ae29035b6d?auto=format&fit=crop&w=1350&q=80",
        caption="Sustainable Logistics in Action 🚚🌿",
        use_column_width=True,
    )

    st.markdown("<p style='text-align:right; color:gray;'>Developed by: Ankit Dixit (Edunet)👨‍💻</p>", unsafe_allow_html=True)

def load_tips():
    st.title("💡 Emission Optimization Tips for Logistics")
    st.markdown(
        """
        Here are some effective strategies to help optimize your carbon footprint during deliveries:
        """
    )
    tips = [
        "🚚 **Optimize routes** to minimize total travel distance and avoid traffic congestion.",
        "⚖️ **Manage cargo weight** effectively; lighter loads improve fuel efficiency.",
        "🌡️ **Monitor weather conditions** to anticipate delays and plan accordingly.",
        "⛽ **Maintain vehicles** regularly for optimal mileage and reduced emissions.",
        "🚦 **Avoid idling** — reduce stop-and-go situations where possible.",
        "🚗 **Use fuel-efficient or alternative fuel vehicles** if available.",
        "📅 **Plan deliveries during off-peak hours** to avoid traffic delays.",
        "📊 **Use data-driven tools** to predict emissions and optimize operations.",
        "🔄 **Combine shipments** to maximize cargo capacity and reduce trips."
    ]
    for tip in tips:
        st.markdown(f"- {tip}")

    st.image(
        "https://images.unsplash.com/photo-1562072544-dc7ecfbb9657?auto=format&fit=crop&w=1350&q=80",
        caption="Efficient and Green Logistics 🌍",
        use_column_width=True,
    )
    st.markdown("---")
    st.markdown("### Additional Resources")
    st.write(
        """
        - [EPA SmartWay Program](https://www.epa.gov/smartway)
        - [International Transport Forum - Sustainable Transport](https://www.itf-oecd.org/sustainable-transport)
        - [Green Freight Europe](https://greenfreighteurope.eu/)
        """
    )

if page == "🏠 Home":
    load_home()
elif page == "⚖️ Carbon Emission Calculator":
    exec(open("main.py", encoding="utf-8").read())
elif page == "🗺️ Route Visualization":
    exec(open("app.py", encoding="utf-8").read())
elif page == "💡 Emission Optimization Tips":
    load_tips()
elif page == "🛠️ Admin Panel":
    exec(open("admin.py", encoding="utf-8").read())
