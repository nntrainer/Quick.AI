# SPDX-License-Identifier: Apache-2.0
#
# Unit tests for Quick.AI. Enabled with `-Denable-test=true`. Requires
# GoogleTest; resolved via pkg-config first, then via the bundled gtest.wrap
# fallback.

gtest_dep = dependency('gtest', main : true, required : false)
if not gtest_dep.found()
  gtest_dep = dependency('gtest_main', required : true,
                         fallback : ['gtest', 'gtest_main_dep'])
endif

test_common_deps = [
  nntrainer_dep,
  nntrainer_ccapi_dep,
  quick_dot_ai_layer_dependencies,
  quick_dot_ai_dep,
  gtest_dep,
  openmp_dep,
]

test_common_inc = quick_dot_ai_inc

# ---------------------------------------------------------------------------
# Tests that do NOT need any model weights. These run on every CI job.
# ---------------------------------------------------------------------------

unittest_llm_util_exe = executable('unittest_llm_util',
  files('unittest_llm_util.cpp'),
  include_directories : test_common_inc,
  dependencies : test_common_deps,
  install : false,
)
test('llm_util', unittest_llm_util_exe, timeout : 60)

unittest_factory_exe = executable('unittest_factory',
  files('unittest_factory.cpp'),
  include_directories : test_common_inc,
  dependencies : test_common_deps,
  install : false,
)
test('factory', unittest_factory_exe, timeout : 60)

unittest_model_config_exe = executable('unittest_model_config',
  files('unittest_model_config.cpp'),
  include_directories : test_common_inc,
  dependencies : test_common_deps,
  install : false,
)
test('model_config', unittest_model_config_exe, timeout : 60)

# ---------------------------------------------------------------------------
# Tests that depend on assets produced by test/scripts/download_qwen3_0.6b.sh.
# They GTEST_SKIP at runtime if the files are absent, so registering them
# unconditionally is safe.
# ---------------------------------------------------------------------------

unittest_tokenizer_exe = executable('unittest_tokenizer',
  files('unittest_tokenizer.cpp'),
  include_directories : test_common_inc,
  dependencies : test_common_deps,
  install : false,
)
test('tokenizer', unittest_tokenizer_exe, timeout : 120,
  workdir : meson.project_source_root(),
)

unittest_causal_lm_api_exe = executable('unittest_causal_lm_api',
  files('unittest_causal_lm_api.cpp'),
  include_directories : test_common_inc,
  dependencies : test_common_deps,
  install : false,
)
test('causal_lm_api', unittest_causal_lm_api_exe, timeout : 600,
  workdir : meson.project_source_root(),
)
