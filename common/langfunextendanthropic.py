"""Language models from Anthropic."""

import base64
import os
from typing import Annotated, Any

import langfun.core as lf
from langfun.core import modalities as lf_modalities
from langfun.core.llms import rest
import pyglove as pg


SUPPORTED_MODELS_AND_SETTINGS = {
    # See https://docs.anthropic.com/claude/docs/models-overview
    # Rate limits from https://docs.anthropic.com/claude/reference/rate-limits
    #     RPM/TPM for Claude-2.1, Claude-2.0, and Claude-Instant-1.2 estimated
    #     as RPM/TPM of the largest-available model (Claude-3-Opus).
    'claude-3-5-sonnet-20240620': pg.Dict(
        max_tokens=4096, rpm=4000, tpm=400000
    ),'claude-3-5-sonnet-latest': pg.Dict(
        max_tokens=4096, rpm=4000, tpm=400000
    ),
    'claude-3-opus-20240229': pg.Dict(max_tokens=4096, rpm=4000, tpm=400000),
    'claude-3-sonnet-20240229': pg.Dict(max_tokens=4096, rpm=4000, tpm=400000),
    'claude-3-haiku-20240307': pg.Dict(max_tokens=4096, rpm=4000, tpm=400000),
    'claude-2.1': pg.Dict(max_tokens=4096, rpm=4000, tpm=400000),
    'claude-2.0': pg.Dict(max_tokens=4096, rpm=4000, tpm=400000),
    'claude-instant-1.2': pg.Dict(max_tokens=4096, rpm=4000, tpm=400000),
}


@lf.use_init_args(['model'])
class Anthropic(rest.REST):
  """Anthropic LLMs (Claude) through REST APIs.

  See https://docs.anthropic.com/claude/reference/messages_post
  """

  model: pg.typing.Annotated[
      pg.typing.Enum(
          pg.MISSING_VALUE, list(SUPPORTED_MODELS_AND_SETTINGS.keys())
      ),
      'The name of the model to use.',
  ]

  multimodal: Annotated[bool, 'Whether this model has multimodal support.'] = (
      True
  )

  api_key: Annotated[
      str | None,
      (
          'API key. If None, the key will be read from environment variable '
          "'ANTHROPIC_API_KEY'."
      ),
  ] = None

  api_endpoint: str = 'https://api.anthropic.com/v1/messages'

  api_version: Annotated[
      str,
      'Anthropic API version.'
  ] = '2023-06-01'

  def _on_bound(self):
    super()._on_bound()
    self._api_key = None

  def _initialize(self):
    api_key = self.api_key or os.environ.get('ANTHROPIC_API_KEY', None)
    if not api_key:
      raise ValueError(
          'Please specify `api_key` during `__init__` or set environment '
          'variable `ANTHROPIC_API_KEY` with your Anthropic API key.'
      )
    self._api_key = api_key

  @property
  def headers(self) -> dict[str, Any]:
    return {
        'x-api-key': self._api_key,
        'anthropic-version': self.api_version,
        'content-type': 'application/json',
    }

  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return self.model

  @property
  def max_concurrency(self) -> int:
    rpm = SUPPORTED_MODELS_AND_SETTINGS[self.model].get('rpm', 0)
    tpm = SUPPORTED_MODELS_AND_SETTINGS[self.model].get('tpm', 0)
    return self.rate_to_max_concurrency(
        requests_per_min=rpm, tokens_per_min=tpm
    )

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
    # Authropic requires `max_tokens` to be specified.
    max_tokens = (
        options.max_tokens
        or SUPPORTED_MODELS_AND_SETTINGS[self.model].max_tokens
    )
    args = dict(
        model=self.model,
        max_tokens=max_tokens,
        stream=False,
    )
    if options.stop:
      args['stop_sequences'] = options.stop
    if options.temperature is not None:
      args['temperature'] = options.temperature
    if options.top_k is not None:
      args['top_k'] = options.top_k
    if options.top_p is not None:
      args['top_p'] = options.top_p
    return args

  def _content_from_message(self, prompt: lf.Message) -> list[dict[str, Any]]:
    """Converts an message to Anthropic's content protocol (list of dicts)."""
    # Refer: https://docs.anthropic.com/claude/reference/messages-examples
    if self.multimodal:
      content = []
      for chunk in prompt.chunk():
        if isinstance(chunk, str):
          item = dict(type='text', text=chunk)
        elif isinstance(chunk, lf_modalities.Image):
          # NOTE(daiyip): Anthropic only support image content instead of URL.
          item = dict(
              type='image',
              source=dict(
                  type='base64',
                  media_type=chunk.mime_type,
                  data=base64.b64encode(chunk.to_bytes()).decode(),
              ),
          )
        else:
          raise ValueError(f'Unsupported modality object: {chunk!r}.')
        content.append(item)
      return content
    else:
      return [dict(type='text', text=prompt.text)]

  def result(self, json: dict[str, Any]) -> lf.LMSamplingResult:
    message = self._message_from_content(json['content'])
    input_tokens = json['usage']['input_tokens']
    output_tokens = json['usage']['output_tokens']
    return lf.LMSamplingResult(
        [lf.LMSample(message)],
        usage=lf.LMSamplingUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )

  def _message_from_content(self, content: list[dict[str, Any]]) -> lf.Message:
    """Converts Anthropic's content protocol to message."""
    # Refer: https://docs.anthropic.com/claude/reference/messages-examples
    return lf.AIMessage.from_chunks(
        [x['text'] for x in content if x['type'] == 'text']
    )