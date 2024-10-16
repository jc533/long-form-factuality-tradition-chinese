import os
from typing import Annotated, Any

import langfun.core as lf
from langfun.core import modalities as lf_modalities
from langfun.core.llms import rest
import pyglove as pg


SUPPORTED_MODELS_AND_SETTINGS = {
    # Refer https://console.groq.com/docs/models
    'llama3-8b-8192': pg.Dict(max_tokens=8192, max_concurrency=16),
    'llama3-70b-8192': pg.Dict(max_tokens=8192, max_concurrency=16),
    'llama2-70b-4096': pg.Dict(max_tokens=4096, max_concurrency=16),
    'mixtral-8x7b-32768': pg.Dict(max_tokens=32768, max_concurrency=16),
    'gemma-7b-it': pg.Dict(max_tokens=8192, max_concurrency=16),
    'llama-3.1-70b-versatile':pg.Dict(max_tokens=8192, max_concurrency=16)
}



@lf.use_init_args(['model'])
class Groq(rest.REST):
  """Groq LLMs through REST APIs (OpenAI compatible).

  See https://platform.openai.com/docs/api-reference/chat
  """

  model: pg.typing.Annotated[
      pg.typing.Enum(
          pg.MISSING_VALUE, list(SUPPORTED_MODELS_AND_SETTINGS.keys())
      ),
      'The name of the model to use.',
  ]

  multimodal: Annotated[bool, 'Whether this model has multimodal support.'] = (
      False
  )

  api_key: Annotated[
      str | None,
      (
          'API key. If None, the key will be read from environment variable '
          "'GROQ_API_KEY'."
      ),
  ] = None

  api_endpoint: str = 'https://api.groq.com/openai/v1/chat/completions'

  def _on_bound(self):
    super()._on_bound()
    self._api_key = None

  def _initialize(self):
    api_key = self.api_key or os.environ.get('GROQ_API_KEY', None)
    if not api_key:
      raise ValueError(
          'Please specify `api_key` during `__init__` or set environment '
          'variable `GROQ_API_KEY` with your Groq API key.'
      )
    self._api_key = api_key

  @property
  def headers(self) -> dict[str, Any]:
    return {
        'Authorization': f'Bearer {self._api_key}',
        'Content-Type': 'application/json',
    }

  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return self.model

  @property
  def max_concurrency(self) -> int:
    return SUPPORTED_MODELS_AND_SETTINGS[self.model].max_concurrency

  def request(
      self,
      prompt: lf.Message,
      sampling_options: lf.LMSamplingOptions
  ) -> dict[str, Any]:
    """Returns the JSON input for a message."""
    request = dict()
    request.update(self._request_args(sampling_options))
    request.update(
        dict(
            messages=[
                dict(role='user', content=self._content_from_message(prompt))
            ]
        )
    )
    return request

  def _request_args(self, options: lf.LMSamplingOptions) -> dict[str, Any]:
    """Returns a dict as request arguments."""
    # `logprobs` and `top_logprobs` flags are not supported on Groq yet.
    args = dict(
        model=self.model,
        n=options.n,
        stream=False,
    )

    if options.temperature is not None:
      args['temperature'] = options.temperature
    if options.max_tokens is not None:
      args['max_tokens'] = options.max_tokens
    if options.top_p is not None:
      args['top_p'] = options.top_p
    if options.stop:
      args['stop'] = options.stop
    return args

  def _content_from_message(self, prompt: lf.Message) -> list[dict[str, Any]]:
    """Converts an message to Groq's content protocol (list of dicts)."""
    # Refer: https://platform.openai.com/docs/api-reference/chat/create
    content = []
    for chunk in prompt.chunk():
      if isinstance(chunk, str):
        item = dict(type='text', text=chunk)
      elif (
          self.multimodal
          and isinstance(chunk, lf_modalities.Image)
          and chunk.uri
      ):
        # NOTE(daiyip): Groq only support image URL.
        item = dict(type='image_url', image_url=chunk.uri)
      else:
        raise ValueError(f'Unsupported modality object: {chunk!r}.')
      content.append(item)
    return content

  def result(self, json: dict[str, Any]) -> lf.LMSamplingResult:
    samples = [
        lf.LMSample(self._message_from_choice(choice), score=0.0)
        for choice in json['choices']
    ]
    usage = json['usage']
    return lf.LMSamplingResult(
        samples,
        usage=lf.LMSamplingUsage(
            prompt_tokens=usage['prompt_tokens'],
            completion_tokens=usage['completion_tokens'],
            total_tokens=usage['total_tokens'],
        ),
    )

  def _message_from_choice(self, choice: dict[str, Any]) -> lf.Message:
    """Converts Groq's content protocol to message."""
    # Refer: https://platform.openai.com/docs/api-reference/chat/create
    content = choice['message']['content']
    if isinstance(content, str):
      return lf.AIMessage(content)
    return lf.AIMessage.from_chunks(
        [x['text'] for x in content if x['type'] == 'text']
    )

@lf.use_init_args(['model'])
class Groq(rest.REST):
  """Groq LLMs through REST APIs (OpenAI compatible).

  See https://platform.openai.com/docs/api-reference/chat
  """

  model: pg.typing.Annotated[
      pg.typing.Enum(
          pg.MISSING_VALUE, list(SUPPORTED_MODELS_AND_SETTINGS.keys())
      ),
      'The name of the model to use.',
  ]

  multimodal: Annotated[bool, 'Whether this model has multimodal support.'] = (
      False
  )

  api_key: Annotated[
      str | None,
      (
          'API key. If None, the key will be read from environment variable '
          "'GROQ_API_KEY'."
      ),
  ] = None

  api_endpoint: str = 'https://api.groq.com/openai/v1/chat/completions'

  def _on_bound(self):
    super()._on_bound()
    self._api_key = None

  def _initialize(self):
    api_key = self.api_key or os.environ.get('GROQ_API_KEY', None)
    if not api_key:
      raise ValueError(
          'Please specify `api_key` during `__init__` or set environment '
          'variable `GROQ_API_KEY` with your Groq API key.'
      )
    self._api_key = api_key

  @property
  def headers(self) -> dict[str, Any]:
    return {
        'Authorization': f'Bearer {self._api_key}',
        'Content-Type': 'application/json',
    }

  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return self.model

  @property
  def max_concurrency(self) -> int:
    return SUPPORTED_MODELS_AND_SETTINGS[self.model].max_concurrency

  def request(
      self,
      prompt: lf.Message,
      sampling_options: lf.LMSamplingOptions
  ) -> dict[str, Any]:
    """Returns the JSON input for a message."""
    request = dict()
    request.update(self._request_args(sampling_options))
    request.update(
        dict(
            messages=[
                dict(role='user', content=self._content_from_message(prompt))
            ]
        )
    )
    return request

  def _request_args(self, options: lf.LMSamplingOptions) -> dict[str, Any]:
    """Returns a dict as request arguments."""
    # `logprobs` and `top_logprobs` flags are not supported on Groq yet.
    args = dict(
        model=self.model,
        n=options.n,
        stream=False,
    )

    if options.temperature is not None:
      args['temperature'] = options.temperature
    if options.max_tokens is not None:
      args['max_tokens'] = options.max_tokens
    if options.top_p is not None:
      args['top_p'] = options.top_p
    if options.stop:
      args['stop'] = options.stop
    return args

  def _content_from_message(self, prompt: lf.Message) -> list[dict[str, Any]]:
    """Converts an message to Groq's content protocol (list of dicts)."""
    # Refer: https://platform.openai.com/docs/api-reference/chat/create
    content = []
    for chunk in prompt.chunk():
      if isinstance(chunk, str):
        item = dict(type='text', text=chunk)
      elif (
          self.multimodal
          and isinstance(chunk, lf_modalities.Image)
          and chunk.uri
      ):
        # NOTE(daiyip): Groq only support image URL.
        item = dict(type='image_url', image_url=chunk.uri)
      else:
        raise ValueError(f'Unsupported modality object: {chunk!r}.')
      content.append(item)
    return content

  def result(self, json: dict[str, Any]) -> lf.LMSamplingResult:
    samples = [
        lf.LMSample(self._message_from_choice(choice), score=0.0)
        for choice in json['choices']
    ]
    usage = json['usage']
    return lf.LMSamplingResult(
        samples,
        usage=lf.LMSamplingUsage(
            prompt_tokens=usage['prompt_tokens'],
            completion_tokens=usage['completion_tokens'],
            total_tokens=usage['total_tokens'],
        ),
    )

  def _message_from_choice(self, choice: dict[str, Any]) -> lf.Message:
    """Converts Groq's content protocol to message."""
    # Refer: https://platform.openai.com/docs/api-reference/chat/create
    content = choice['message']['content']
    if isinstance(content, str):
      return lf.AIMessage(content)
    return lf.AIMessage.from_chunks(
        [x['text'] for x in content if x['type'] == 'text']
    )
  


