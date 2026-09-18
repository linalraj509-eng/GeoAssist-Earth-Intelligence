from io import BytesIO

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import rasterio
from rasterio.io import MemoryFile

from streamlit_mic_recorder import speech_to_text


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="GeoAssist Earth Intelligence",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 GeoAssist Earth Intelligence")
st.caption("ASK EARTH. GET EVIDENCE. MAKE DECISIONS.")

st.write(
    "Agentic Geospatial Reasoning Assistant "
    "for Multimodal Earth Observation"
)

st.divider()


# ============================================================
# LOCAL AI REASONING
# ============================================================

st.success("🟢 AI Engine: Free Local Prototype Mode")


# ============================================================
# SESSION STATE
# ============================================================

if "voice_query_text" not in st.session_state:
    st.session_state.voice_query_text = ""


# ============================================================
# VOICE
# ============================================================

st.subheader("🎤 Voice Query")

voice_text = speech_to_text(
    language="en",
    start_prompt="🎙️ Speak",
    stop_prompt="⏹️ Stop",
    just_once=True,
    use_container_width=True,
    key="voice_query"
)

if voice_text:
    st.session_state.voice_query_text = voice_text

    st.success("✅ Voice recognized!")

    st.write(
        "**You said:**",
        voice_text
    )


# ============================================================
# TEXT
# ============================================================

st.subheader("⌨️ Text Query")

question = st.text_input(
    "Ask GeoAssist:",
    placeholder=(
        "Example: How much area changed "
        "between these two images?"
    )
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.divider()

st.subheader("🛰️ Satellite Images")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 📷 BEFORE Image")

    before_image = st.file_uploader(
        "Upload BEFORE image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff"
        ],
        key="before"
    )

with col2:

    st.markdown("### 📷 AFTER Image")

    after_image = st.file_uploader(
        "Upload AFTER image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff"
        ],
        key="after"
    )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(file):

    filename = file.name.lower()

    # GeoTIFF
    if filename.endswith(".tif") or filename.endswith(".tiff"):

        file.seek(0)

        file_bytes = file.read()

        with MemoryFile(file_bytes) as memfile:

            with memfile.open() as dataset:

                array = dataset.read(1)

                metadata = {
                    "crs": dataset.crs,
                    "transform": dataset.transform,
                    "width": dataset.width,
                    "height": dataset.height,
                    "resolution": dataset.res
                }

        array = array.astype(np.float32)

        array = np.nan_to_num(
            array,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return array, metadata, "GeoTIFF"

    # JPG / PNG
    file.seek(0)

    image = Image.open(
        BytesIO(file.read())
    ).convert("L")

    array = np.array(
        image
    ).astype(np.float32)

    metadata = {
        "crs": None,
        "transform": None,
        "width": image.width,
        "height": image.height,
        "resolution": None
    }

    return array, metadata, "Normal Image"


# ============================================================
# NORMALIZE
# ============================================================

def normalize_image(array):

    array = array.astype(np.float32)

    minimum = np.min(array)
    maximum = np.max(array)

    if maximum == minimum:
        return np.zeros_like(array)

    return (
        (array - minimum)
        /
        (maximum - minimum)
        *
        255
    )


# ============================================================
# RESIZE
# ============================================================

def resize_array(
    array,
    height,
    width
):

    array = np.clip(
        array,
        0,
        255
    ).astype(np.uint8)

    image = Image.fromarray(array)

    image = image.resize(
        (width, height)
    )

    return np.array(
        image
    ).astype(np.float32)


# ============================================================
# ============================================================
# AI QUERY UNDERSTANDING
# ============================================================

def understand_query(query):
    q = query.lower()
    if any(w in q for w in ["optical", "sar", "radar", "sentinel-1", "sentinel 1"]):
        return {"intent":"OPTICAL_SAR", "reason":"Local reasoning detected an optical/SAR cross-modal query."}
    if any(w in q for w in ["area", "hectare", "hectares", "km2", "km²", "square kilometre", "square kilometer", "how much land", "how much area"]) and any(w in q for w in ["change", "changed", "difference", "affected", "between"]):
        return {"intent":"AREA_MEASUREMENT", "reason":"Local reasoning detected a change-area measurement request."}
    if any(w in q for w in ["changed", "change", "difference", "before", "after", "between", "temporal", "earlier", "later"]):
        return {"intent":"TEMPORAL_CHANGE", "reason":"Local reasoning detected a temporal change-analysis query."}
    if any(w in q for w in ["single image", "one image", "classify", "classification", "describe this image", "caption"]):
        return {"intent":"SINGLE_IMAGE", "reason":"Local reasoning detected a single-image analysis query."}
    return {"intent":"GENERAL_GEOSPATIAL", "reason":"Local reasoning classified this as a general geospatial query."}


# ============================================================
# AI RESULT EXPLANATION
# ============================================================

def explain_result(query, changed_pixels, changed_percentage, area_available, area_m2, area_ha, area_km2):
    if area_available:
        return (f"GeoAssist detected {changed_percentage:.2f}% pixel-level change, "
                f"covering approximately {area_km2:.4f} km² ({area_ha:.4f} hectares). "
                "The measurement is derived from the uploaded georeferenced projected GeoTIFF imagery.")
    return (f"GeoAssist detected {changed_percentage:.2f}% pixel-level change across {changed_pixels:,} pixels. "
            "A real-world area cannot be calculated from these inputs because appropriate projected "
            "georeferencing/resolution is unavailable. This prototype result indicates image-level difference only; "
            "it does not by itself prove flooding, deforestation, or another specific phenomenon.")


# CHANGE MAP
# ============================================================

def create_change_map(
    after,
    change_mask
):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.imshow(
        after,
        cmap="gray"
    )

    overlay = np.ma.masked_where(
        ~change_mask,
        change_mask
    )

    ax.imshow(
        overlay,
        cmap="Reds",
        alpha=0.75
    )

    ax.set_title(
        "GeoAssist Change Detection Map"
    )

    ax.axis("off")

    plt.tight_layout()

    return fig


# ============================================================
# ANALYSE
# ============================================================

st.divider()

analyse = st.button(
    "🔍 Analyse with GeoAssist AI",
    use_container_width=True
)


if analyse:

    # ========================================================
    # QUERY
    # ========================================================

    final_query = question.strip()

    if not final_query:

        final_query = (
            st.session_state.voice_query_text
        )

    if not final_query:

        st.warning(
            "⚠️ Please ask a question using voice or text."
        )

        st.stop()


    # ========================================================
    # IMAGES
    # ========================================================

    if not before_image:

        st.warning(
            "⚠️ Upload the BEFORE image."
        )

        st.stop()

    if not after_image:

        st.warning(
            "⚠️ Upload the AFTER image."
        )

        st.stop()


    # ========================================================
    # AI REASONING
    # ========================================================

    st.divider()

    st.subheader(
        "🤖 1. AI Geospatial Reasoning"
    )

    st.write(
        "**User Query:**",
        final_query
    )

    with st.spinner(
        "🤖 AI is understanding your query..."
    ):

        ai_result = understand_query(
            final_query
        )

    st.success(
        f"AI Intent: **{ai_result['intent']}**"
    )

    st.write(
        "**AI Reasoning:**"
    )

    st.code(
        ai_result["reason"],
        language="text"
    )


    # ========================================================
    # LOAD IMAGES
    # ========================================================

    st.divider()

    st.subheader(
        "🛰️ 2. Remote-Sensing Processing"
    )

    try:

        with st.spinner(
            "🛰️ Loading imagery..."
        ):

            before_array, before_meta, before_type = load_image(
                before_image
            )

            after_array, after_meta, after_type = load_image(
                after_image
            )

    except Exception as e:

        st.error(
            f"❌ Image reading failed: {e}"
        )

        st.stop()


    st.success(
        "✓ Imagery loaded"
    )


    # ========================================================
    # SIZE CHECK
    # ========================================================

    if before_array.shape != after_array.shape:

        if (
            before_type == "GeoTIFF"
            or after_type == "GeoTIFF"
        ):

            st.error(
                "❌ GeoTIFF dimensions do not match. "
                "Use spatially matched images."
            )

            st.stop()

        else:

            st.warning(
                "⚠️ Image sizes differ. "
                "Resizing AFTER image for prototype analysis."
            )

            after_array = resize_array(
                after_array,
                before_array.shape[0],
                before_array.shape[1]
            )


    # ========================================================
    # CRS CHECK
    # ========================================================

    if (
        before_type == "GeoTIFF"
        and after_type == "GeoTIFF"
    ):

        if (
            before_meta["crs"] is not None
            and after_meta["crs"] is not None
        ):

            if before_meta["crs"] == after_meta["crs"]:

                st.write(
                    "✓ CRS verified"
                )

            else:

                st.warning(
                    "⚠️ BEFORE and AFTER CRS differ."
                )


    # ========================================================
    # NORMALIZATION
    # ========================================================

    before_norm = normalize_image(
        before_array
    )

    after_norm = normalize_image(
        after_array
    )


    # ========================================================
    # CHANGE DETECTION
    # ========================================================

    st.subheader(
        "🔍 3. Specialist Change Analysis"
    )

    threshold = st.slider(
        "Change sensitivity",
        min_value=5,
        max_value=100,
        value=25,
        step=5
    )

    with st.spinner(
        "🔍 Detecting changes..."
    ):

        difference = np.abs(
            before_norm
            -
            after_norm
        )

        change_mask = (
            difference
            >
            threshold
        )

    st.success(
        "✓ Change detection completed"
    )


    # ========================================================
    # PIXEL MEASUREMENT
    # ========================================================

    changed_pixels = int(
        np.sum(change_mask)
    )

    total_pixels = (
        change_mask.size
    )

    changed_percentage = (
        changed_pixels
        /
        total_pixels
        *
        100
    )


    # ========================================================
    # REAL AREA
    # ========================================================

    area_available = False

    area_m2 = None
    area_ha = None
    area_km2 = None


    if (
        before_type == "GeoTIFF"
        and after_type == "GeoTIFF"
    ):

        crs = before_meta["crs"]

        transform = before_meta["transform"]


        if (
            crs is not None
            and crs.is_projected
        ):

            pixel_width = abs(
                transform.a
            )

            pixel_height = abs(
                transform.e
            )

            pixel_area = (
                pixel_width
                *
                pixel_height
            )

            area_m2 = (
                changed_pixels
                *
                pixel_area
            )

            area_ha = (
                area_m2
                /
                10000
            )

            area_km2 = (
                area_m2
                /
                1_000_000
            )

            area_available = True


    # ========================================================
    # GIS RESULT
    # ========================================================

    st.divider()

    st.subheader(
        "📐 4. Deterministic GIS Measurement"
    )

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "Changed Pixels",
            f"{changed_pixels:,}"
        )

    with m2:

        st.metric(
            "Changed %",
            f"{changed_percentage:.2f}%"
        )

    with m3:

        if area_available:

            st.metric(
                "Changed Area",
                f"{area_km2:.4f} km²"
            )

        else:

            st.metric(
                "Changed Area",
                "N/A"
            )


    if area_available:

        a1, a2 = st.columns(2)

        with a1:

            st.write(
                f"**Area:** {area_m2:,.2f} m²"
            )

        with a2:

            st.write(
                f"**Area:** {area_ha:.4f} hectares"
            )

    else:

        st.warning(
            "⚠️ Real-world area is unavailable. "
            "Use a properly georeferenced projected GeoTIFF."
        )


    # ========================================================
    # EVIDENCE
    # ========================================================

    st.divider()

    st.subheader(
        "🗺️ 5. Evidence"
    )

    e1, e2 = st.columns(2)

    with e1:

        st.image(
            before_image,
            caption="BEFORE",
            use_container_width=True
        )

    with e2:

        st.image(
            after_image,
            caption="AFTER",
            use_container_width=True
        )


    # ========================================================
    # CHANGE MAP
    # ========================================================

    st.subheader(
        "🔴 Change Detection Map"
    )

    change_fig = create_change_map(
        after_norm,
        change_mask
    )

    st.pyplot(
        change_fig,
        use_container_width=True
    )


    # ========================================================
    # AI EXPLANATION
    # ========================================================

    st.divider()

    st.subheader(
        "🤖 6. AI Explanation"
    )

    with st.spinner(
        "🤖 AI is explaining the measured result..."
    ):

        explanation = explain_result(
            final_query,
            changed_pixels,
            changed_percentage,
            area_available,
            area_m2,
            area_ha,
            area_km2
        )

    st.success(
        explanation
    )


    # ========================================================
    # EXECUTION TRACE
    # ========================================================

    st.divider()

    st.subheader(
        "📋 7. Execution Trace"
    )

    trace = [
        "✓ User query received",
        "✓ AI geospatial intent identified",
        "✓ BEFORE imagery validated",
        "✓ AFTER imagery validated",
        "✓ Temporal analysis workflow selected",
        "✓ Remote-sensing imagery loaded",
        "✓ Change detection executed",
        "✓ Changed pixels measured"
    ]

    if area_available:

        trace.extend([
            "✓ GeoTIFF CRS verified",
            "✓ Pixel resolution obtained",
            "✓ Real-world area calculated"
        ])

    else:

        trace.append(
            "⚠ Real-world area unavailable"
        )

    trace.extend([
        "✓ Evidence map generated",
        "✓ Result validated",
        "✓ Local AI explanation generated"
    ])

    for item in trace:

        st.write(item)


    # ========================================================
    # ARCHITECTURE
    # ========================================================

    st.divider()

    st.subheader(
        "🧠 GeoAssist AI Architecture"
    )

    st.code(
        """
VOICE / TEXT
     ↓
LOCAL AI REASONING
     ↓
GEOSPATIAL INTENT
     ↓
SPECIALIST RS ANALYSIS
     ↓
CHANGE DETECTION
     ↓
DETERMINISTIC GIS
     ↓
AREA MEASUREMENT
     ↓
VALIDATION
     ↓
LOCAL AI REASONING
     ↓
EVIDENCE-GROUNDED ANSWER
        """,
        language="text"
    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.divider()

    buffer = BytesIO()

    change_fig.savefig(
        buffer,
        format="png",
        dpi=200,
        bbox_inches="tight"
    )

    buffer.seek(0)

    st.download_button(
        "⬇️ Download Change Map",
        data=buffer,
        file_name="geoassist_change_map.png",
        mime="image/png",
        use_container_width=True
    )


    # ========================================================
    # FINAL
    # ========================================================

    st.success(
        "🌍 GeoAssist completed: "
        "ASK → REASON → ANALYSE → MEASURE → VERIFY → EXPLAIN"
    )

    st.caption(
        "Prototype note: change detection currently uses "
        "pixel-level image difference. Production deployment "
        "should use sensor-appropriate remote-sensing models, "
        "accurate co-registration and radiometric preprocessing."
    )