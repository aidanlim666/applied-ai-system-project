import streamlit as st
from pawpal_system import Owner, Pet, Task, Scheduler
from rag_advisor import RAGAdvisor, DISCLAIMER


@st.cache_resource
def get_advisor() -> RAGAdvisor:
    """Cached across reruns — builds the TF-IDF index once per session."""
    return RAGAdvisor()

st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="centered")

st.title("🐾 PawPal+")

st.markdown(
    """
Welcome to the PawPal+ starter app.

This file is intentionally thin. It gives you a working Streamlit app so you can start quickly,
but **it does not implement the project logic**. Your job is to design the system and build it.

Use this app as your interactive demo once your backend classes/functions exist.
"""
)

with st.expander("Scenario", expanded=True):
    st.markdown(
        """
**PawPal+** is a pet care planning assistant. It helps a pet owner plan care tasks
for their pet(s) based on constraints like time, priority, and preferences.

You will design and implement the scheduling logic and connect it to this Streamlit UI.
"""
    )

with st.expander("What you need to build", expanded=True):
    st.markdown(
        """
At minimum, your system should:
- Represent pet care tasks (what needs to happen, how long it takes, priority)
- Represent the pet and the owner (basic info and preferences)
- Build a plan/schedule for a day that chooses and orders tasks based on constraints
- Explain the plan (why each task was chosen and when it happens)
"""
    )

st.divider()

st.subheader("Owner & Pet Info")
owner_name = st.text_input("Owner name", value="Jordan")
available_minutes = st.number_input("Available time (minutes)", min_value=10, max_value=480, value=120)
day_start = st.text_input("Day start time (HH:MM)", value="08:00")
pet_name = st.text_input("Pet name", value="Mochi")
species = st.selectbox("Species", ["dog", "cat", "other"])

# Initialize Pet in session state once — guards against overwrite on every rerun
if "pet" not in st.session_state:
    st.session_state.pet = Pet(name=pet_name, species=species)

st.markdown("### Tasks")
st.caption("Add tasks below — each one is saved as a Task object on your Pet.")

col1, col2 = st.columns(2)
with col1:
    task_title = st.text_input("Task title", value="Morning walk")
with col2:
    category = st.selectbox("Category", ["walk", "feeding", "meds", "grooming", "enrichment"])

st.markdown("##### AI-assisted defaults (RAG)")
st.caption(
    "Retrieves relevant care-guideline snippets and proposes a duration/priority default. "
    "You always review and can accept, edit, or ignore it."
)
if st.button("🔎 Suggest duration & priority"):
    st.session_state.last_suggestion = get_advisor().suggest(
        species=species, category=category, task_title=task_title
    )

if "last_suggestion" in st.session_state:
    sug = st.session_state.last_suggestion
    if sug.source == "llm":
        st.info(
            f"**AI suggestion (Claude):** {sug.duration_minutes} min, priority: **{sug.priority}** "
            f"(confidence: {sug.confidence:.2f})\n\n"
            f"{sug.rationale}\n\n"
            f"Source: {sug.citation}\n\n"
            f"_{DISCLAIMER}_"
        )
    elif sug.source == "heuristic":
        st.info(
            f"**AI suggestion (offline heuristic mode):** {sug.duration_minutes} min, "
            f"priority: **{sug.priority}** (confidence: {sug.confidence:.2f})\n\n"
            f"{sug.rationale}\n\n"
            f"Source: {sug.citation}\n\n"
            f"Set `ANTHROPIC_API_KEY` to get live Claude-generated suggestions instead. "
            f"_{DISCLAIMER}_"
        )
    else:
        st.warning(
            f"**Rule-based default** (suggestion did not pass validation): "
            f"{sug.duration_minutes} min, priority: **{sug.priority}**\n\n{sug.rationale}"
        )
    acc_col, dismiss_col = st.columns(2)
    with acc_col:
        if st.button("Use this suggestion"):
            st.session_state.duration_input = sug.duration_minutes
            st.session_state.priority_input = sug.priority
            st.session_state.applied_citation = sug.citation
            st.session_state.applied_values = (sug.duration_minutes, sug.priority)
            st.rerun()
    with dismiss_col:
        if st.button("Dismiss"):
            del st.session_state["last_suggestion"]
            st.rerun()

col3, col4 = st.columns(2)
with col3:
    duration = st.number_input(
        "Duration (minutes)", min_value=1, max_value=240, value=20, key="duration_input"
    )
with col4:
    priority = st.selectbox(
        "Priority", ["low", "medium", "high"], index=2, key="priority_input"
    )

fixed_time = st.text_input("Fixed start time (optional, e.g. 08:00)", value="")

if st.button("Add task"):
    applied_values = st.session_state.get("applied_values")
    citation = None
    if applied_values == (int(duration), priority):
        citation = st.session_state.get("applied_citation")

    task = Task(
        title=task_title,
        duration_minutes=int(duration),
        priority=priority,
        category=category,
        fixed_time=fixed_time.strip() if fixed_time.strip() else None,
        ai_citation=citation,
    )
    st.session_state.pet.add_task(task)
    st.success(f"Added: {task_title}")

scheduler = Scheduler()

current_tasks = st.session_state.pet.get_tasks()
if current_tasks:
    st.write("Current tasks (chronological order — flexible tasks last):")
    st.table([
        {
            "Title": t.title,
            "Duration (min)": t.duration_minutes,
            "Priority": t.priority,
            "Category": t.category,
            "Fixed time": t.fixed_time or "—",
        }
        for t in scheduler.sort_by_time(current_tasks)
    ])
else:
    st.info("No tasks yet. Add one above.")

st.divider()

st.subheader("Build Schedule")
st.caption("Builds a daily plan from the tasks above using your Scheduler class.")

if st.button("Generate schedule"):
    owner = Owner(
        name=owner_name,
        available_minutes=int(available_minutes),
        day_start=day_start,
    )
    pet = st.session_state.pet
    pet.name = pet_name
    pet.species = species
    owner.add_pet(pet)

    schedule = scheduler.schedule_day(pet, owner)

    st.markdown("#### Today's Plan")
    st.text(schedule.get_summary())
    st.markdown("#### Reasoning")
    st.text(schedule.explain())

    conflict_warnings = scheduler.check_conflicts([schedule])
    if conflict_warnings:
        st.markdown("#### Conflict Warnings")
        for warning in conflict_warnings:
            st.warning(warning)
    else:
        st.success("No time conflicts detected.")
