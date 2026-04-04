from enum import Enum
from io import StringIO

class PromptType(Enum):
    Alpaca = 0
    Phi_3 = 1

def format_input(instruction, input, prompt_type: PromptType = PromptType.Alpaca):
    content = StringIO()

    match prompt_type:
        case PromptType.Alpaca:
            content.write(
                f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request."
                f"\n\n### Instruction:\n{instruction}"
            )
            content.write( f"\n\n### Input:\n{input}" if input else "" )
        case PromptType.Phi_3:
            content.write(
                f"<|user|>\n{instruction.rstrip('.')}: '{input}'\n\n"
            )

    return content.getvalue()


def format_instruction(entry, prompt_type: PromptType = PromptType.Alpaca):
    '''
    Prompt Template Construct
    :param entry:
    :param prompt_type:
    :return:
    '''
    instruction = StringIO()

    match prompt_type:
        case PromptType.Alpaca:

            instruction.write(
                f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request."
                f"\n\n### Instruction:\n{entry['instruction']}"
            )

            instruction.write( f"\n\n### Input:\n{entry['input']}" if entry["input"] else "" )

            instruction.write( f"\n\n### Response:\n{entry['output']}" )

        case PromptType.Phi_3:
            instruction.write(
                f"<|user|>\n{entry['instruction'].rstrip('.')}: '{entry['input']}'\n\n"
                f"<|assistant|>\n{entry['output']}\n"
            )
        case _:
            raise Exception("Invalid Prompt!")

    return instruction.getvalue()