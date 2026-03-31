from streamlit.testing.v1 import AppTest


def test_logged_out_landing_page_renders():
    at = AppTest.from_file("streamlit_app.py")
    at.run(timeout=15)

    assert not at.exception
    assert [item.value for item in at.title] == ["ICU → HDU Step-Down Readiness"]
    assert any(
        "Sign in from the sidebar" in item.value
        for item in at.info
    )
