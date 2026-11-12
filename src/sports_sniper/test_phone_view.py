"""The phone layout: tables as cards, with the data escaped and only real sportsbook links kept."""
import pandas as pd
from streamlit.testing.v1 import AppTest

from sports_sniper import phone_view


def frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"Leg": "Al <script>Star</script> — Over 20.5 Points", "Game": "Away @ Home", "Odds": "+110", "Model view": "Disagrees (46%)",
         "Projection": "-", "Bet": "https://sportsbook.fanduel.com/addToBetslip?marketId=1&selectionId=2"},
        {"Leg": "Bo Mate — Under 5.5 Assists", "Game": "Away @ Home", "Odds": "-120", "Model view": None, "Projection": "4.8",
         "Bet": "javascript:alert(1)"},
        {"Leg": "Cy Away — Over 1.5 Threes", "Game": None, "Odds": "+150", "Model view": "-", "Projection": "2.1", "Bet": None},
    ])


def test_cards_show_each_row_escaped_with_only_https_links():
    out = phone_view.cards_html(frame(), sub=["Game"])
    assert out.count('class="sn-card"') == 3 and "<script>" not in out and "&lt;script&gt;" in out
    assert out.count('class="sn-open"') == 1 and "javascript:" not in out                    # one real link, the others dropped
    assert 'href="https://sportsbook.fanduel.com/addToBetslip?marketId=1&amp;selectionId=2"' in out and 'rel="noopener noreferrer"' in out
    assert out.count('class="sn-sub"') == 2 and out.count("<span>Projection</span>") == 2     # empty values are left out
    assert '<div class="sn-wide"><span>Model view</span><b>Disagrees (46%)</b>' in out        # a long value gets its own row
    assert "<span>Bet</span>" not in out and "<span>Leg</span>" not in out and "<span>Game</span>" not in out


def test_cards_keep_chosen_columns_in_order_and_stop_at_the_limit():
    out = phone_view.cards_html(frame(), title="Game", keep=["Projection", "Odds", "Missing"], link=None)
    last = out[out.rindex('class="sn-card"'):]
    assert last.index("<span>Projection</span>") < last.index("<span>Odds</span>") and "Model view" not in out and "sn-open" not in out
    assert '<div class="sn-title">Away @ Home</div>' in out and '<div class="sn-title">-</div>' in out
    many = pd.DataFrame({"Name": [f"row {n}" for n in range(50)], "Value": range(50)})
    assert phone_view.cards_html(many).count("sn-card") == phone_view.CARD_LIMIT
    assert phone_view.cards_html(many.iloc[0:0]) == ""


def page(phone: bool):
    def script(phone):
        import pandas as pd
        import streamlit as st
        from sports_sniper import phone_view
        st.session_state.setdefault(phone_view.STATE_KEY, phone)
        on = phone_view.setup()
        phone_view.table(pd.DataFrame({"Name": [f"row {n}" for n in range(40)], "Value": range(40)}))
        st.write(f"phone={on}")
    return AppTest.from_function(script, args=(phone,)).run()


def test_the_table_is_a_grid_on_a_wide_screen_and_cards_in_the_phone_layout():
    wide = page(False)
    assert not wide.exception and len(wide.dataframe) == 1 and wide.sidebar.toggle[0].label == "Phone layout"
    assert not wide.sidebar.toggle[0].value and not phone_view.detected()                     # no browser to read here: not a phone
    narrow = page(True)
    assert not narrow.exception and len(narrow.dataframe) == 0
    assert any("sn-card" in m.value for m in narrow.markdown) and "first 30 of 40" in narrow.caption[-1].value
