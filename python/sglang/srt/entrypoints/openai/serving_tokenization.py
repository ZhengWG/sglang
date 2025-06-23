from typing import Any, Dict, List, Optional, Union

from fastapi import Request
from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints.openai.protocol import (
    TokenizeRequest,
    TokenizeChatRequest,
    TokenizeCompletionRequest,
    TokenizeResponse,
    ErrorResponse,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.managers.io_struct import GenerateReqInput


class OpenAIServingTokenization(OpenAIServingBase):
    """Handler for v1/tokenize requests"""

    def _request_id_prefix(self) -> str:
        return "tokn-"

    def _convert_to_internal_request(
        self,
        request: TokenizeRequest,
    ) -> tuple[GenerateReqInput, TokenizeRequest]:
        """Convert OpenAI tokenize request to internal format"""
        return None, request

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: TokenizeRequest,
        raw_request: Request,
    ) -> Union[TokenizeResponse, ErrorResponse, ORJSONResponse]:
        """Handle the tokenization request"""
        if isinstance(request, TokenizeChatRequest):
            # process messages
            openai_compatible_messages = []
            for message in request.messages:
                if message.content is None:
                    message.content = ""
                msg_dict = message.dict()
                if isinstance(msg_dict.get("content"), list):
                    for chunk in msg_dict["content"]:
                        if isinstance(chunk, dict) and chunk.get("type") == "text":
                            new_msg = msg_dict.copy()
                            new_msg["content"] = chunk["text"]
                            new_msg = {
                                k: v for k, v in new_msg.items() if v is not None
                            }
                            openai_compatible_messages.append(new_msg)
                else:
                    msg_dict = {k: v for k, v in msg_dict.items() if v is not None}
                    openai_compatible_messages.append(msg_dict)
            prompt_ids = self.tokenizer_manager.tokenizer.apply_chat_template(
                openai_compatible_messages,
                tokenize=True,
                add_generation_prompt=request.add_generation_prompt,
                continue_final_message=request.continue_final_message,
                **(
                    request.chat_template_kwargs
                    if request.chat_template_kwargs
                    else {}
                ),
            )
        else:
            prompt_ids = self.tokenizer_manager.tokenizer.encode(
                request.prompt,
                add_special_tokens=request.add_special_tokens,
            )

        return TokenizeResponse(tokens=prompt_ids,
                                count=len(prompt_ids),
                                max_model_len=self.tokenizer_manager.context_len)
