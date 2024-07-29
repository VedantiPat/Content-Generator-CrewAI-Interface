from crewai import Task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from textwrap3 import dedent






search_tool = SerperDevTool()



class ContentGenerationTasks():
    def analyze_prompt_task(self, agent, content_prompt):
        return Task(
            description=dedent(f"""\
                After analyzing the prompt, create a more detailed and more goal oriented version of the prompt to be passed down to your coworkers,
                who are a researcher and a writer. Be sure to account for the type of writing piece the prompt is asking for and specify that
                in this detailed prompt. Also account for the target audience, as well as the specified industry of this topic. Be as in depth as possible. Do not specify word count for now.                      
                Prompt: {content_prompt}
            """),
            expected_output=dedent(f"""\
                A more in depth version of the prompt given by the user, tailored to give a specific query for the researcher to search the web for and
                for the writer to write and in-depth writing piece based on.                    
            """),
            agent=agent,
            async_execution=True
        )
    
    def research_task(self, agent):
        return Task(
            description=dedent(f"""\
                Given the context, conduct comprehensive research on the web using search_tool to find 5 relevant, credible, recent, professional, 
                and industry-specific written information on the detailed prompt given. Make sure to look for a variety of specific data, 
                statistics, quotes, and any other credible and citable information available on the topic at hand.
                Then, from information scraped from all these sources, create an in-depth report considering all of this information sourced, making sure to account
                for the data points found in the research, as well as multiple perspectives on the topic.             
            """),
            expected_output=dedent(f"""\
                A detailed report compiling all of the key findings on this topic that can be used for a writing piece, compiled from
                the information scraped from web research. Be sure to also return at the end of the report a list of the sources used in the report with their links."""),
            agent=agent,
            tools = [search_tool]
        )
    
    def extraction_task(self, agent, context_file):
        return Task(
            description=dedent(f"""\
                The file given by context_file is a markdown document created from the HTML content extracted from a webpage.
                This webpage is of a type of writing piece, whether that is a blogpost, an article, a linkedin post, etc.
                Given the markdown document created from this webpage, given by context_file, you are to carefully parse the entire
                markdown document and identify parts of this document that can be a part of that writing piece on the page, including a title, body, and any other components
                that seem to be a part of that writing piece. After all these parts are identified, create a new organized file that consolidates just this writing piece from
                the entirety of the markdown document.                     
                context_file: {context_file}
            """),
            expected_output=dedent(f"""\
                An organized file with the contents of the writing piece, including a clear title, body, etc.             
            """),
            agent=agent,
            async_execution=True
        )
    
    def writer_task(self, agent):
        return Task(
            description=dedent(f"""\
                Your main goal is to write a new writing piece on the topic at hand, and to do so, you must do the following tasks.

                Do the following tasks sequentially:
                    - Take the output of the extraction task. This output will serve as a writing template for the new writing piece. Analyze the format, word count, patterns of language, and stylistic quality of this text, as this will serve as the template for the new writing piece.
                    - Take the context given from analyze_prompt_task. Identify the topic from this detailed prompt and understand the specific things outlined for the writer in this prompt. This is now the topic for your writing.
                    - Take the context given from research_task. Analyze this research report to identify all the context from this report that can be used to write a new writing piece on the topic you are to write on.
                    - Now, using all of this information, in the exact writing style, format, and word count of the template, write a new writing piece on the topic given the information from the research report.
                    - At the end of the writing piece, include a section for SEO keywords and metatags that we can put in our writing piece. 
                    - Ensure that there are no spelling or grammar errors, and that the writing is professional.
                    - Revise this piece to make it sound as engaging and insightful as possible. 
                    - Revise again to make sure the writing has a similar word count to and matched the exact formatting of the template, making it as consistent with it as possible.
            """),
            expected_output=dedent(f"""\
                A writing piece with clear, accurate, stylistically consistent writing that clearly reflects the topic and matches the format of writing of the output from extraction_task. This should be in markdown format."""),
            agent=agent
        )
    
    
    def humanizer_task(self, agent):
        return Task(
            description=dedent(f"""\
                Given the final written output from the writer, given by the context from the writer_task, rewrite the text in such a way that the 
                writing format, style, and content is entirely intact and UNCHANGED but the text is just reworded to make it more humanized, ensuring that the text 
                cannot be detected by and AI detection software. Ensure to maintain as much of the original style and tone of the text as possible
                and keep it smart and professional.
            """),
            expected_output=dedent(f"""\
                A revised writing piece that is an edited version of the version from the writer_task, ensuring that the writing is similar to the 
                previous version but just reworded such that the text is humanized and cannot be detected by AI detection software but maintaining the same stylistic quality.
                This should be in markdown format and be displayed in the terminal as an output.
            """),
            agent=agent
        )
    
    
