"""
🍌 CHILL ETHAKKA — ABSURD BOTANICAL METROLOGY & ORACLE MODULE
Completely useless, delightfully over-engineered interactive templates:
1. പഴ ജാതകം (The Banana Horoscope & Destiny Oracle)
2. ഏറോഡൈനാമിക്സ് ഫ്ലൈറ്റ് സിമുലേറ്റർ (Aerodynamic Boomerang Flight Simulator)
"""

import json
from typing import Dict, Any


def generate_jathakam_data(curve_score: float, category: str, arc_length: float, chord: float) -> Dict[str, Any]:
    """Generates hilarious Kerala astrology reading based on banana curvature."""
    # Constellation / Nakshathram
    if curve_score < 5.0:
        star_name = "അശ്വതി നേന്ത്രൻ (Aswathi Nendran)"
        star_title = "The Rigid Orthogonalist"
        rashi = "മേടം (Aries)"
        dosham_title = "അമിത നേർരേഖാ ദോഷം (Hyper-Orthogonal Affliction)"
        dosham_desc = "വളയാൻ വിസമ്മതിക്കുന്ന പ്രകൃതം. റൂളർ പോലെ നേരെ നിൽക്കുന്നതിനാൽ പരമ്പരാഗത ഫ്രൂട്ട് ബൗളുകളിൽ മറ്റു വളഞ്ഞ പഴങ്ങളുമായി ഒത്തുപോകാൻ അല്പം ബുദ്ധിമുട്ടാണ്. വളയ്ക്കാൻ ശ്രമിച്ചാൽ പൊട്ടാൻ സാധ്യതയുണ്ട്."
        yogam = "റൂളർ മഹാ യോഗം (The Divine Protractor): ഫിസിക്സ് ലാബിലെ അളവുകോലായി പുനർജനിക്കാൻ സാധ്യത."
        pariharam = "ഒരു വളഞ്ഞ റോബസ്റ്റ പഴത്തിന്റെ കൂടെ 10 മിനിറ്റ് ഇരുത്തി സഹിഷ്ണുത വളർത്തുക."
        temperament = "വിട്ടുവീഴ്ചയില്ലാത്തവൻ (Unbending & Stoic)"
        luck_direction = "വടക്ക് (ആപ്പിൾ ബോക്സിന് അഭിമുഖമായി)"
    elif curve_score <= 15.0:
        star_name = "രോഹിണി റോബസ്റ്റ (Rohini Robusta)"
        star_title = "The Harmonious Crescent"
        rashi = "ഇടവം (Taurus)"
        dosham_title = "സൗമ്യ വക്ര ഭാഗ്യം (Harmonious Golden Arc)"
        dosham_desc = "അത്യുത്തമമായ സാത്വിക വക്രത. സമൂഹത്തിലും ചായക്കടകളിലും ഉന്നത പദവി. ഡൈനിംഗ് ടേബിളിൽ ആരുടെയും മനം കവരുന്ന വടിവൊത്ത ശരീരം."
        yogam = "പഴംപൊരി മഹാ യോഗം (Grand Fritter Union): വൈകിട്ട് 4:30-ന് ചൂടുള്ള വെളിച്ചെണ്ണയിൽ സ്നാനം ചെയ്ത് സായൂജ്യമടയും."
        pariharam = "പ്രത്യേക പരിഹാരം ആവശ്യമില്ല. തണുത്ത ഫ്രിഡ്ജിൽ വെക്കാതിരുന്നാൽ ധാരാളം."
        temperament = "ശാന്തശീലൻ, സർവ്വപ്രിയൻ (Sweet & Balanced)"
        luck_direction = "കിഴക്ക് (ചായപ്പാത്രത്തിന് അഭിമുഖമായി)"
    elif curve_score <= 25.0:
        star_name = "തിരുവാതിര പൂവൻ (Thiruvathira Poovan)"
        star_title = "The Royal Arch"
        rashi = "ചിങ്ങം (Leo)"
        dosham_title = "വക്ര മധ്യ ദോഷം (The Crescent Charm)"
        dosham_desc = "തീവ്രമായ വളവ്. സഞ്ചികളിൽ വെച്ചാൽ മറ്റുള്ള സാധനങ്ങളുടെ മേൽ ആധിപത്യം സ്ഥാപിക്കും. സ്വന്തം ഭംഗിയിൽ അമിത അഹംഭാവം."
        yogam = "പഴം നുറുക്ക് യോഗം: നെയ്യിൽ വഴറ്റി ഏലയ്ക്കാപ്പൊടി വിതറി രാജകീയമായി ഭോജനം ചെയ്യപ്പെടും."
        pariharam = "വൈകുന്നേരങ്ങളിൽ സുജാതയുടെ പാട്ടുകൾ കേൾപ്പിക്കുന്നത് മനസ്സിന് ശാന്തി നൽകും."
        temperament = "കലാകാരൻ, നാടകീയ പ്രകൃതം (Artistic & Flamboyant)"
        luck_direction = "തെക്ക്-കിഴക്ക് (ഡൈനിംഗ് ഹാൾ)"
    else:
        star_name = "ഭരണി ഏത്തക്ക (Bharani Boomerang)"
        star_title = "The Wild Crescent Boomerang"
        rashi = "വൃശ്ചികം (Scorpio)"
        dosham_title = "ചൊവ്വാ വക്ര ദോഷം (Boomerang Malefic Affliction)"
        dosham_desc = "അതിതീവ്ര വക്രത! മേശപ്പുറത്തുനിന്ന് ഉരുട്ടി വിട്ടാൽ തിരികെ കയ്യിലേക്ക് തന്നെ പറന്നുവരാൻ 86% സാധ്യത. തനിയെ ഇരിക്കാൻ ഇഷ്ടപ്പെടുന്ന ഏകാന്ത വിപ്ലവകാരി."
        yogam = "ബൂമറാങ് യോഗം: എയറോഡൈനാമിക് ഭ്രമണപഥത്തിൽ പ്രവേശിച്ച് സീലിംഗ് ഫാനിൽ തട്ടി താഴെ വീഴാൻ നേരിയ യോഗം."
        pariharam = "ഉടൻതന്നെ ഒരു കപ്പ് ചൂടുള്ള നാടൻ ചായയുടെ അടുത്തു വെച്ച് മനസ്സ് തണുപ്പിക്കുക."
        temperament = "വിപ്ലവകാരി, സാഹസികൻ (Wild, Kinetic & Unpredictable)"
        luck_direction = "ആകാശ മാർഗ്ഗം (Skyward)"

    # House contents for the 12-box Kerala Jathakam
    chakra_houses = {
        "meenam": "വ്യാഴം<br><small>പഴുപ്പ്</small>",
        "medam": f"ലഗ്നം ({curve_score:.1f}%)" if curve_score < 7 else "രവി<br><small>പ്രകാശം</small>",
        "edavam": "ശുക്രൻ<br><small>മധുരം</small>",
        "midhunam": "ബുധൻ<br><small>വടിവ്</small>",
        "karkkidakam": f"ലഗ്നം ({curve_score:.1f}%)" if 7 <= curve_score < 14 else "ചന്ദ്രൻ<br><small>തണുപ്പ്</small>",
        "chingam": "സൂര്യൻ<br><small>തങ്കനിറം</small>",
        "kanni": "കേതു<br><small>കറുത്ത പുള്ളി</small>",
        "thulaam": f"ലഗ്നം ({curve_score:.1f}%)" if 14 <= curve_score < 22 else "ശനി<br><small>ഫ്രിഡ്ജ് ദോഷം</small>",
        "vrischikam": "ചൊവ്വ<br><small>വക്രത</small>",
        "dhanu": "ഗുളികൻ<br><small>ഈച്ച ശല്യം</small>",
        "makaram": f"ലഗ്നം ({curve_score:.1f}%)" if curve_score >= 22 else "രാഹു<br><small>അതിശീതം</small>",
        "kumbham": "പ്ലുട്ടോ<br><small>സൂക്ഷ്മം</small>",
    }

    return {
        "star_name": star_name,
        "star_title": star_title,
        "rashi": rashi,
        "dosham_title": dosham_title,
        "dosham_desc": dosham_desc,
        "yogam": yogam,
        "pariharam": pariharam,
        "temperament": temperament,
        "luck_direction": luck_direction,
        "chakra": chakra_houses,
        "score": curve_score,
        "arc_length": arc_length,
        "chord": chord,
    }


def render_absurd_interactive_hub(result_data: Dict[str, Any]) -> str:
    """
    Renders an interactive web application container containing:
    1. പഴ ജാതകം (The Banana Horoscope & Astrological Chart)
    2. ഏറോഡൈനാമിക്സ് ഫ്ലൈറ്റ് സിമുലേറ്റർ (Boomerang Flight Physics)
    """
    curve_score = float(result_data.get("curve_score", 12.0))
    category = result_data.get("category", "curved")
    path_len = float(result_data.get("path_length", 300.0))
    chord_len = float(result_data.get("chord_distance", 280.0))
    max_defl = float(result_data.get("max_deflection", 30.0))

    jathakam = generate_jathakam_data(curve_score, category, path_len, chord_len)
    
    # Calculate flight physics
    cl_lift = min(1.85, 0.25 + (curve_score / 20.0) * 0.9)
    return_chance = min(98.5, max(12.0, 95.0 - abs(curve_score - 18.0) * 3.2))
    flight_dist = max(5.0, 12.0 + (chord_len / 40.0) * (curve_score / 10.0))
    ceiling_risk = min(96, int(curve_score * 3.1 + 15))

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;1,400&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@500;700&family=Manjari:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-ivory: #F8F5EC;
            --surface-white: #FFFFFF;
            --surface-card: #FAF8F2;
            --border-gold: #D4AF37;
            --border-subtle: #E8E2D5;
            --gold-gradient: linear-gradient(135deg, #B8860B 0%, #E6CA65 50%, #996515 100%);
            --text-charcoal: #1C1917;
            --text-secondary: #57534E;
            --accent-red: #991B1B;
            --font-sans: 'Plus Jakarta Sans', sans-serif;
            --font-serif: 'Playfair Display', serif;
            --font-mono: 'JetBrains Mono', monospace;
            --font-malayalam: 'Manjari', sans-serif;
            --font-cinzel: 'Cinzel', serif;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
        }}

        body {{
            background: transparent;
            font-family: var(--font-sans);
            color: var(--text-charcoal);
            padding: 6px 2px;
        }}

        .hub-container {{
            background: #FCFAF6;
            border: 2px solid #E5DFD1;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            overflow: hidden;
            margin: 0 auto;
            max-width: 1040px;
        }}

        /* Tab Switcher Header */
        .hub-tabs {{
            display: flex;
            background: #EFE9DC;
            border-bottom: 2px solid #E2D9C7;
            padding: 8px 12px 0 12px;
            gap: 8px;
        }}

        .tab-btn {{
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            padding: 13px 18px;
            font-family: var(--font-sans);
            font-size: 0.92rem;
            font-weight: 700;
            color: var(--text-secondary);
            background: #E4DCCB;
            border: 1px solid #D8CEB9;
            border-bottom: none;
            border-radius: 12px 12px 0 0;
            cursor: pointer;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        }}

        .tab-btn:hover {{
            background: #F5EFE4;
            color: var(--text-charcoal);
        }}

        .tab-btn.active {{
            background: #FCFAF6;
            color: #854D0E;
            border-color: #D4AF37;
            border-bottom: 2px solid #FCFAF6;
            margin-bottom: -2px;
            box-shadow: 0 -4px 12px rgba(212, 175, 55, 0.15);
        }}

        .tab-content {{
            display: none;
            padding: 22px;
            animation: fadeIn 0.3s ease-in-out forwards;
        }}

        .tab-content.active {{
            display: block;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* ========================================================
           TAB 1: PAZHAM JATHAKAM (ASTROLOGICAL CHART & ORACLE)
           ======================================================== */
        .jathakam-wrapper {{
            background: #FFFDF9;
            border: 2px solid #DECBB1;
            border-radius: 14px;
            padding: 24px;
            position: relative;
            background-image: radial-gradient(#F3EBDD 1px, transparent 1px);
            background-size: 20px 20px;
            box-shadow: inset 0 0 20px rgba(212, 175, 55, 0.08);
        }}

        .jathakam-corner {{
            position: absolute;
            width: 28px;
            height: 28px;
            border: 3px solid #C49A45;
        }}
        .corner-tl {{ top: 8px; left: 8px; border-right: none; border-bottom: none; }}
        .corner-tr {{ top: 8px; right: 8px; border-left: none; border-bottom: none; }}
        .corner-bl {{ bottom: 8px; left: 8px; border-right: none; border-top: none; }}
        .corner-br {{ bottom: 8px; right: 8px; border-left: none; border-top: none; }}

        .jathakam-header {{
            text-align: center;
            margin-bottom: 20px;
            border-bottom: 1px dashed #D6C2A5;
            padding-bottom: 14px;
        }}

        .jathakam-emblem {{
            font-size: 2.2rem;
            margin-bottom: 4px;
            display: inline-block;
            filter: drop-shadow(0 2px 8px rgba(212, 175, 55, 0.4));
        }}

        .jathakam-bureau {{
            font-family: var(--font-cinzel);
            font-size: 0.8rem;
            letter-spacing: 0.18em;
            color: #9A6E24;
            font-weight: 700;
            text-transform: uppercase;
        }}

        .jathakam-mal-title {{
            font-family: var(--font-malayalam);
            font-size: 1.95rem;
            font-weight: 700;
            color: #78350F;
            margin: 3px 0 2px 0;
            letter-spacing: 0.02em;
        }}

        .jathakam-sub {{
            font-family: var(--font-serif);
            font-style: italic;
            color: var(--text-secondary);
            font-size: 0.92rem;
        }}

        .jathakam-grid-layout {{
            display: grid;
            grid-template-columns: 310px 1fr;
            gap: 22px;
            align-items: start;
        }}

        /* Traditional 12-House Kerala Chakra (4x4 Grid) */
        .chakra-container {{
            background: #FAF5EA;
            border: 3px double #C49A45;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        }}

        .chakra-title-bar {{
            text-align: center;
            font-family: var(--font-cinzel);
            font-size: 0.72rem;
            font-weight: 800;
            color: #854D0E;
            letter-spacing: 0.1em;
            margin-bottom: 8px;
        }}

        .rashi-table {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            grid-template-rows: repeat(4, 58px);
            border: 2px solid #854D0E;
            background: #FFFDF9;
        }}

        .rashi-cell {{
            border: 1px solid #C49A45;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            font-family: var(--font-malayalam);
            font-size: 0.72rem;
            font-weight: 700;
            color: #451A03;
            padding: 2px;
            line-height: 1.15;
            position: relative;
        }}

        .rashi-cell small {{
            font-size: 0.58rem;
            color: #9A3412;
            font-weight: 600;
        }}

        /* Center 2x2 merged sanctum */
        .rashi-center {{
            grid-column: 2 / 4;
            grid-row: 2 / 4;
            border: 2px solid #854D0E;
            background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            box-shadow: inset 0 0 10px rgba(212, 175, 55, 0.3);
        }}

        .rashi-center-icon {{
            font-size: 1.6rem;
            line-height: 1;
        }}

        .rashi-center-label {{
            font-family: var(--font-malayalam);
            font-size: 0.85rem;
            font-weight: 700;
            color: #78350F;
            margin-top: 2px;
        }}

        /* Prophecy Cards Right Side */
        .prophecy-list {{
            display: flex;
            flex-direction: column;
            gap: 11px;
        }}

        .prophecy-card {{
            background: #FFFFFF;
            border: 1px solid #EAE2D3;
            border-left: 4px solid #C49A45;
            border-radius: 8px;
            padding: 11px 15px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.02);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}

        .prophecy-card:hover {{
            transform: translateX(4px);
            box-shadow: 0 4px 12px rgba(196, 154, 69, 0.12);
        }}

        .prophecy-label {{
            font-family: var(--font-cinzel);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            color: #A16207;
            text-transform: uppercase;
            margin-bottom: 2px;
        }}

        .prophecy-val-highlight {{
            font-family: var(--font-malayalam);
            font-size: 1.1rem;
            font-weight: 700;
            color: #451A03;
            line-height: 1.3;
        }}

        .prophecy-val-sub {{
            font-size: 0.84rem;
            color: #57534E;
            margin-top: 3px;
            line-height: 1.4;
        }}

        /* Interactive Blessing Button */
        .blessing-box {{
            margin-top: 18px;
            padding-top: 14px;
            border-top: 1px dashed #DECBB1;
            text-align: center;
        }}

        .blessing-btn {{
            background: linear-gradient(135deg, #B45309 0%, #D97706 50%, #B45309 100%);
            color: #FFFFFF;
            font-family: var(--font-malayalam);
            font-size: 0.98rem;
            font-weight: 700;
            padding: 11px 26px;
            border: 2px solid #FDE68A;
            border-radius: 50px;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(180, 83, 9, 0.3);
            transition: all 0.25s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }}

        .blessing-btn:hover {{
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 6px 20px rgba(180, 83, 9, 0.45);
            background: linear-gradient(135deg, #92400E 0%, #B45309 100%);
        }}

        .blessing-display {{
            display: none;
            margin-top: 14px;
            background: #FEF3C7;
            border: 1px solid #F59E0B;
            border-radius: 8px;
            padding: 12px 18px;
            font-family: var(--font-malayalam);
            font-size: 0.98rem;
            font-weight: 700;
            color: #78350F;
            animation: popIn 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
        }}

        @keyframes popIn {{
            from {{ opacity: 0; transform: scale(0.9); }}
            to {{ opacity: 1; transform: scale(1); }}
        }}

        /* ========================================================
           TAB 2: AERODYNAMIC BOOMERANG FLIGHT SIMULATOR
           ======================================================== */
        .flight-wrapper {{
            background: #FFFFFF;
            border: 2px solid #E5DFD1;
            border-radius: 14px;
            padding: 22px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.03);
        }}

        .flight-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid #EAE2D3;
        }}

        .flight-title-group h3 {{
            font-family: var(--font-sans);
            font-size: 1.22rem;
            font-weight: 800;
            color: var(--text-charcoal);
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .flight-title-group p {{
            font-size: 0.84rem;
            color: var(--text-secondary);
            margin-top: 2px;
        }}

        .flight-sky-stage {{
            position: relative;
            height: 260px;
            background: linear-gradient(180deg, #E0F2FE 0%, #BAE6FD 65%, #F0FDF4 100%);
            border: 2px solid #7DD3FC;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: inset 0 4px 16px rgba(0,0,0,0.04);
            margin-bottom: 16px;
        }}

        /* Animated Clouds */
        .sky-cloud {{
            position: absolute;
            background: rgba(255, 255, 255, 0.85);
            border-radius: 50px;
            filter: blur(1px);
        }}
        .cloud-1 {{ width: 110px; height: 32px; top: 30px; left: 15%; animation: floatCloud 24s linear infinite; }}
        .cloud-2 {{ width: 140px; height: 38px; top: 75px; left: 65%; animation: floatCloud 32s linear infinite reverse; }}

        @keyframes floatCloud {{
            0% {{ transform: translateX(0); }}
            50% {{ transform: translateX(40px); }}
            100% {{ transform: translateX(0); }}
        }}

        /* The Flying Banana Element */
        #simulatedBanana {{
            position: absolute;
            bottom: 35px;
            left: 55px;
            font-size: 2.8rem;
            line-height: 1;
            transform-origin: center center;
            z-index: 10;
            filter: drop-shadow(0 4px 8px rgba(0,0,0,0.25));
            transition: filter 0.2s ease;
        }}

        /* Target Landing Zone */
        .landing-pad {{
            position: absolute;
            bottom: 15px;
            left: 40px;
            padding: 4px 14px;
            background: rgba(34, 197, 94, 0.2);
            border: 1.5px dashed #16A34A;
            border-radius: 20px;
            font-family: var(--font-mono);
            font-size: 0.65rem;
            font-weight: 700;
            color: #15803D;
        }}

        .wind-indicator {{
            position: absolute;
            top: 12px;
            right: 14px;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #BAE6FD;
            padding: 6px 12px;
            border-radius: 20px;
            font-family: var(--font-mono);
            font-size: 0.72rem;
            font-weight: 700;
            color: #0369A1;
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        /* Flight Stats Telemetry Row */
        .telemetry-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 18px;
        }}

        .telemetry-card {{
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 10px 12px;
            text-align: center;
        }}

        .telemetry-label {{
            font-family: var(--font-mono);
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: #64748B;
            text-transform: uppercase;
        }}

        .telemetry-value {{
            font-family: var(--font-mono);
            font-size: 1.22rem;
            font-weight: 800;
            color: #0F172A;
            margin-top: 4px;
        }}

        .telemetry-unit {{
            font-size: 0.75rem;
            color: #94A3B8;
            margin-left: 2px;
        }}

        /* Launch Action Bar */
        .launch-bar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #F1F5F9;
            border-radius: 10px;
            padding: 12px 18px;
            gap: 16px;
        }}

        .launch-btn {{
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            padding: 12px 28px;
            font-family: var(--font-sans);
            font-size: 0.95rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .launch-btn:hover {{
            background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
        }}

        .flight-status-text {{
            font-family: var(--font-mono);
            font-size: 0.8rem;
            font-weight: 700;
            color: #334155;
        }}

        @media (max-width: 820px) {{
            .jathakam-grid-layout {{
                grid-template-columns: 1fr;
            }}
            .chakra-container {{
                max-width: 320px;
                margin: 0 auto;
            }}
            .telemetry-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            .launch-bar {{
                flex-direction: column;
                text-align: center;
            }}
        }}
    </style>
    </head>
    <body>

    <div class="hub-container">
        <!-- Tab Navigation Bar -->
        <div class="hub-tabs">
            <button class="tab-btn active" id="tabBtnJathakam" onclick="switchTab('jathakam')">
                <span>🔮</span>
                <span>പഴ ജാതകം (BANANA JATHAKAM & ORACLE)</span>
            </button>
            <button class="tab-btn" id="tabBtnFlight" onclick="switchTab('flight')">
                <span>🪃</span>
                <span>ഏറോഡൈനാമിക്സ് (BOOMERANG FLIGHT TEST)</span>
            </button>
        </div>

        <!-- TAB 1 CONTENT: PAZHAM JATHAKAM -->
        <div class="tab-content active" id="tabContentJathakam">
            <div class="jathakam-wrapper">
                <div class="jathakam-corner corner-tl"></div>
                <div class="jathakam-corner corner-tr"></div>
                <div class="jathakam-corner corner-bl"></div>
                <div class="jathakam-corner corner-br"></div>

                <div class="jathakam-header">
                    <div class="jathakam-emblem">🍌 🪔</div>
                    <div class="jathakam-bureau">★ KERALA BOTANICAL ASTROLOGICAL COUNCIL ★</div>
                    <h2 class="jathakam-mal-title">അഖിലലോക പഴ ജാതകം</h2>
                    <p class="jathakam-sub">“Curvature is Destiny: Revealing the Cosmic Karma of Specimen #{abs(hash(str(curve_score))) % 90000 + 10000}”</p>
                </div>

                <div class="jathakam-grid-layout">
                    <!-- Traditional 12-House Kerala Chakra (4x4 Grid) -->
                    <div class="chakra-container">
                        <div class="chakra-title-bar">★ രാശി ചക്രം (CHOSEN HOUSE) ★</div>
                        <div class="rashi-table">
                            <!-- Row 1 -->
                            <div class="rashi-cell">{jathakam['chakra']['meenam']}</div>
                            <div class="rashi-cell">{jathakam['chakra']['medam']}</div>
                            <div class="rashi-cell">{jathakam['chakra']['edavam']}</div>
                            <div class="rashi-cell">{jathakam['chakra']['midhunam']}</div>
                            <!-- Row 2 -->
                            <div class="rashi-cell">{jathakam['chakra']['kumbham']}</div>
                            <!-- Center 2x2 Sanctum -->
                            <div class="rashi-center">
                                <div class="rashi-center-icon">🍌</div>
                                <div class="rashi-center-label">പഴ ജാതകം</div>
                            </div>
                            <div class="rashi-cell">{jathakam['chakra']['karkkidakam']}</div>
                            <!-- Row 3 -->
                            <div class="rashi-cell">{jathakam['chakra']['makaram']}</div>
                            <div class="rashi-cell">{jathakam['chakra']['chingam']}</div>
                            <!-- Row 4 -->
                            <div class="rashi-cell">{jathakam['chakra']['dhanu']}</div>
                            <div class="rashi-cell">{jathakam['chakra']['vrischikam']}</div>
                            <div class="rashi-cell">{jathakam['chakra']['thulaam']}</div>
                            <div class="rashi-cell">{jathakam['chakra']['kanni']}</div>
                        </div>
                    </div>

                    <!-- Astrological Prophecies -->
                    <div class="prophecy-list">
                        <div class="prophecy-card">
                            <div class="prophecy-label">ജന്മ നക്ഷത്രവും രാശിയും (BIRTH CONSTELLATION)</div>
                            <div class="prophecy-val-highlight">{jathakam['star_name']}</div>
                            <div class="prophecy-val-sub">രാശി: <b>{jathakam['rashi']}</b> | സ്വഭാവം: <i>{jathakam['temperament']}</i></div>
                        </div>

                        <div class="prophecy-card">
                            <div class="prophecy-label">വക്രതാ ദോഷം (CURVATURE AFFLICTION / DOSHAM)</div>
                            <div class="prophecy-val-highlight">{jathakam['dosham_title']}</div>
                            <div class="prophecy-val-sub">{jathakam['dosham_desc']}</div>
                        </div>

                        <div class="prophecy-card">
                            <div class="prophecy-label">പരമ യോഗം (DIVINE FATE / CULINARY DESTINY)</div>
                            <div class="prophecy-val-highlight">{jathakam['yogam']}</div>
                            <div class="prophecy-val-sub">ഭാഗ്യ ദിശ: <b>{jathakam['luck_direction']}</b></div>
                        </div>

                        <div class="prophecy-card">
                            <div class="prophecy-label">പരിഹാരം (SACRED REMEDY / PARIHARAM)</div>
                            <div class="prophecy-val-highlight" style="font-size: 0.95rem; color: #9A3412;">{jathakam['pariharam']}</div>
                        </div>
                    </div>
                </div>

                <!-- Interactive Blessing Trigger -->
                <div class="blessing-box">
                    <button class="blessing-btn" onclick="triggerBlessing()">
                        <span>✨</span>
                        <span>അനുഗ്രഹം വാങ്ങുക (RECEIVE COSMIC BLESSING)</span>
                        <span>✨</span>
                    </button>
                    <div class="blessing-display" id="blessingText">
                        🌸 “സർവ്വ പഴ സമ്പൽ സമൃദ്ധിരസ്തു! ഈ പഴത്തിന് ഒരു പോറൽ പോലും ഏൽക്കാതെ ചായക്കടയിൽ പരമപദം പ്രാപിക്കട്ടെ!” 🍌🙏
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 2 CONTENT: BOOMERANG FLIGHT SIMULATOR -->
        <div class="tab-content" id="tabContentFlight">
            <div class="flight-wrapper">
                <div class="flight-header">
                    <div class="flight-title-group">
                        <h3><span>🪃</span> Aerodynamic Boomerang Simulation</h3>
                        <p>Simulating ballistic trajectory based on curvature coefficient ({curve_score:.2f}%)</p>
                    </div>
                    <div class="wind-indicator">
                        <span>🍃 Crosswind: 3.2 kt NE</span>
                    </div>
                </div>

                <!-- Sky Canvas -->
                <div class="flight-sky-stage" id="skyStage">
                    <div class="sky-cloud cloud-1"></div>
                    <div class="sky-cloud cloud-2"></div>

                    <!-- Flying Banana -->
                    <div id="simulatedBanana">🍌</div>

                    <!-- Launch / Landing Pad -->
                    <div class="landing-pad">
                        <span>🎯 LAUNCH PAD A-1</span>
                    </div>
                </div>

                <!-- Flight Telemetry Readouts -->
                <div class="telemetry-grid">
                    <div class="telemetry-card">
                        <div class="telemetry-label">BOOMERANG RETURN CHANCE</div>
                        <div class="telemetry-value" style="color: #2563EB;">{return_chance:.1f}<span class="telemetry-unit">%</span></div>
                    </div>
                    <div class="telemetry-card">
                        <div class="telemetry-label">LIFT COEFFICIENT (CL)</div>
                        <div class="telemetry-value" style="color: #059669;">{cl_lift:.2f}</div>
                    </div>
                    <div class="telemetry-card">
                        <div class="telemetry-label">MAX FLIGHT RANGE</div>
                        <div class="telemetry-value" style="color: #D97706;">{flight_dist:.1f}<span class="telemetry-unit">m</span></div>
                    </div>
                    <div class="telemetry-card">
                        <div class="telemetry-label">CEILING FAN COLLISION RISK</div>
                        <div class="telemetry-value" style="color: #DC2626;">{ceiling_risk}<span class="telemetry-unit">%</span></div>
                    </div>
                </div>

                <!-- Interactive Launch Bar -->
                <div class="launch-bar">
                    <div class="flight-status-text" id="flightStatus">
                        STATUS: READY ON LAUNCH PAD. SPECIMEN ARMED.
                    </div>
                    <button class="launch-btn" id="launchBtn" onclick="launchBananaFlight()">
                        <span>🚀</span>
                        <span>LAUNCH BANANA FLIGHT</span>
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Tab switching logic
        function switchTab(tab) {{
            const tabJathakam = document.getElementById('tabContentJathakam');
            const tabFlight = document.getElementById('tabContentFlight');
            const btnJathakam = document.getElementById('tabBtnJathakam');
            const btnFlight = document.getElementById('tabBtnFlight');

            if (tab === 'jathakam') {{
                tabJathakam.classList.add('active');
                tabFlight.classList.remove('active');
                btnJathakam.classList.add('active');
                btnFlight.classList.remove('active');
            }} else {{
                tabFlight.classList.add('active');
                tabJathakam.classList.remove('active');
                btnFlight.classList.add('active');
                btnJathakam.classList.remove('active');
            }}
        }}

        // Audio synthesis for funny temple bell / flight whoosh
        function playChime() {{
            try {{
                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.type = 'sine';
                osc.frequency.setValueAtTime(587.33, ctx.currentTime); // D5
                osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.3); // A5
                gain.gain.setValueAtTime(0.3, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 1.2);
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.start();
                osc.stop(ctx.currentTime + 1.2);
            }} catch(e) {{}}
        }}

        function playWhoosh() {{
            try {{
                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.type = 'triangle';
                osc.frequency.setValueAtTime(220, ctx.currentTime);
                osc.frequency.exponentialRampToValueAtTime(660, ctx.currentTime + 0.5);
                osc.frequency.exponentialRampToValueAtTime(180, ctx.currentTime + 1.4);
                gain.gain.setValueAtTime(0.25, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 1.5);
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.start();
                osc.stop(ctx.currentTime + 1.5);
            }} catch(e) {{}}
        }}

        // Blessing action
        function triggerBlessing() {{
            playChime();
            const display = document.getElementById('blessingText');
            display.style.display = 'block';
            display.style.animation = 'none';
            void display.offsetWidth; // trigger reflow
            display.style.animation = 'popIn 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards';
        }}

        // Boomerang Flight Animation
        let isFlying = false;
        function launchBananaFlight() {{
            if (isFlying) return;
            isFlying = true;

            const banana = document.getElementById('simulatedBanana');
            const status = document.getElementById('flightStatus');
            const btn = document.getElementById('launchBtn');

            playWhoosh();
            btn.disabled = true;
            btn.style.opacity = '0.6';
            status.innerHTML = 'FLIGHT ENGAGED: ROTATING ALONG CURVATURE VECTOR...';
            status.style.color = '#2563EB';

            const startTime = performance.now();
            const duration = 2400; // ms

            function animateFlight(currentTime) {{
                const elapsed = currentTime - startTime;
                const progress = Math.min(1, elapsed / duration);

                // Parametric Boomerang Flight Path
                const angle = progress * Math.PI * 2;
                const radiusX = 260;
                const radiusY = 140;

                // X moves out then loops back
                const x = 55 + (1 - Math.cos(angle)) * (radiusX / 2);
                // Y arcs up high then descends back
                const y = 35 + Math.sin(progress * Math.PI) * radiusY;
                // Spin rotation based on curve
                const rotation = progress * 1440 * ({curve_score} > 15 ? 1.4 : 1.0);

                banana.style.left = x + 'px';
                banana.style.bottom = y + 'px';
                banana.style.transform = `rotate(${{rotation}}deg)`;

                if (progress < 1) {{
                    requestAnimationFrame(animateFlight);
                }} else {{
                    // Landed
                    banana.style.left = '55px';
                    banana.style.bottom = '35px';
                    banana.style.transform = 'rotate(0deg)';
                    status.innerHTML = 'FLIGHT COMPLETE: TOUCHDOWN ON PAD! BOOMERANG EFFECT: 100% SUCCESS.';
                    status.style.color = '#15803D';
                    btn.disabled = false;
                    btn.style.opacity = '1';
                    isFlying = false;
                }}
            }}

            requestAnimationFrame(animateFlight);
        }}
    </script>
    </body>
    </html>
    """
    return html
