from tmunan.api.pydantic_models import TextInstructions, SequencePrompt
from tmunan.common.event import Event


class TextGenerator:

    def __init__(self):
        self.instructions = TextInstructions()
        self.on_prompt_ready = Event()

    def start(self):
        pass

    def set_instructions(self, instructions: TextInstructions):
        self.instructions = instructions

    def push_text(self, text: str):

        # generate prompt from text
        text_prompt = SequencePrompt(text=text,
                                     start_weight=self.instructions.start_weight,
                                     end_weight=self.instructions.end_weight)

        # fire event
        print(f'Firing on_prompt_ready with: {text_prompt}')
        self.on_prompt_ready.notify(text_prompt)

    def stop(self):
        pass
