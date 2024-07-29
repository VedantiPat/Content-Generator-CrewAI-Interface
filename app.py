import streamlit as st
from crew_vp import Crew
from tasks import ContentGenerationTasks
from agents import ContentGenerationAgents
from tools import Extraction_Tools
import os
os.environ["OPENAI_MODEL_NAME"]="gpt-4o-mini"
from st_copy_to_clipboard import st_copy_to_clipboard

class ContentGeneratorUI:

    def generate_content(self, prompt, link, openai_key, serper_key):
        
        
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["SERPER_API_KEY"] = serper_key

        content_prompt = prompt
        context = link
        context_file = Extraction_Tools.url_to_md(context)

        tasks = ContentGenerationTasks()
        agents = ContentGenerationAgents()

        extraction_agent = agents.extraction_agent()
        prompt_analyst_agent = agents.prompt_analyst_agent()
        researcher_agent = agents.researcher_agent()
        writer_agent = agents.writer_agent()
        humanizer_agent = agents.humanizer_agent()

        extraction_task = tasks.extraction_task(extraction_agent, context_file)
        analyze_prompt_task = tasks.analyze_prompt_task(prompt_analyst_agent, content_prompt)
        researcher_task = tasks.research_task(researcher_agent)
        writer_task = tasks.writer_task(writer_agent)
        humanizer_task = tasks.humanizer_task(humanizer_agent)

        researcher_task.context = [analyze_prompt_task]
        writer_task.context = [extraction_task, analyze_prompt_task, researcher_task]
        humanizer_task.context = [writer_task]

        
        content_generation_crew = Crew(

            agent = [
                extraction_agent,
                prompt_analyst_agent,
                researcher_agent,
                writer_agent,
                humanizer_agent
            ],
            tasks = [
                extraction_task,
                analyze_prompt_task,
                researcher_task,
                writer_task,
                humanizer_task
            ],
        )

        result = content_generation_crew.kickoff()
        output = humanizer_task.output.raw_output

        return output
    
    def content_generation(self):
        if st.session_state.generating:
            st.session_state.content = self.generate_content(
                st.session_state.prompt, st.session_state.link, st.session_state.openai, st.session_state.serper
            )
            with st.container():
                st.write("Content generated successfully!")
                st_copy_to_clipboard(str(st.session_state.content))
                st.markdown(st.session_state.content)
            st.session_state.generating = False
    
    def sidebar(self):
        with st.sidebar:
            st.title("Written Content Generator  ✍️")

            st.write(
                """
                To get started, enter a prompt for the new content you would like to be generated, specifying type of content (blogpost, linkedIn post, etc.), as well as any word count limitations or other details. \n
                Additionally, provide a url to a writing piece you would like to be used for stylistic/formatting context! \n
                """
            )

            st.text_area("Prompt", key="prompt", placeholder="Write a blogpost about...")

            st.text_input("Link", key="link", placeholder="https://www.coforge.com/blog...")
            
            urlgpt = "https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key"
            st.text_input("OpenAI API Key (click [here](%s) for more)" % urlgpt, key="openai", type="password")

            urlserper = "https://serper.dev/api-key"
            st.text_input("SerperDev API Key (click [here](%s) for more)" % urlserper, key="serper", type="password")

            if st.button("Generate!"):
                st.session_state.generating = True

    
    def render(self):
        st.set_page_config(page_title = "Written Content Generator", page_icon="✍️")

        if "prompt" not in st.session_state:
            st.session_state.prompt = ""

        if "link" not in st.session_state:
            st.session_state.link = ""

        if "openai" not in st.session_state:
            st.session_state.openai = ""
        
        if "serper" not in st.session_state:
            st.session_state.serper = ""

        if "content" not in st.session_state:
            st.session_state.content = ""

        if "generating" not in st.session_state:
            st.session_state.generating = False

        self.sidebar()
        self.content_generation()

        


if __name__ == "__main__":
    ContentGeneratorUI().render()
