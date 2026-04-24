// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_factory.cpp
 * @date   24 Apr 2026
 * @brief  GoogleTest unit tests for quick_dot_ai::Factory. The factory is a
 *         singleton string->creator map, so we register throw-away keys
 *         (prefixed with "UT/") to avoid colliding with real model
 *         registrations that happen at library init time.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <sstream>

#include "json.hpp"
#include <factory.h>

using json = nlohmann::json;

namespace {

TEST(FactoryTest, CreateReturnsNullptrForUnknownKey) {
  json cfg = json::object();
  json generation_cfg = json::object();
  json nntr_cfg = json::object();

  auto &factory = quick_dot_ai::Factory::Instance();
  auto result =
    factory.create("UT/DoesNotExist", cfg, generation_cfg, nntr_cfg);

  EXPECT_EQ(result, nullptr);
}

TEST(FactoryTest, RegisteredCreatorIsInvoked) {
  auto &factory = quick_dot_ai::Factory::Instance();

  bool creator_invoked = false;
  factory.registerModel("UT/SpyCreator",
                        [&creator_invoked](json & /*cfg*/, json & /*gen*/,
                                           json & /*nntr*/)
                          -> std::unique_ptr<quick_dot_ai::Transformer> {
                          creator_invoked = true;
                          return nullptr;
                        });

  json cfg = json::object();
  json gen = json::object();
  json nntr = json::object();
  auto result = factory.create("UT/SpyCreator", cfg, gen, nntr);

  EXPECT_TRUE(creator_invoked);
  EXPECT_EQ(result, nullptr); // creator returned nullptr on purpose
}

TEST(FactoryTest, LastRegisteredCreatorWinsForSameKey) {
  auto &factory = quick_dot_ai::Factory::Instance();

  int which = 0;
  factory.registerModel("UT/Overridable", [&which](json &, json &, json &) {
    which = 1;
    return std::unique_ptr<quick_dot_ai::Transformer>(nullptr);
  });
  factory.registerModel("UT/Overridable", [&which](json &, json &, json &) {
    which = 2;
    return std::unique_ptr<quick_dot_ai::Transformer>(nullptr);
  });

  json cfg = json::object();
  json gen = json::object();
  json nntr = json::object();
  (void)factory.create("UT/Overridable", cfg, gen, nntr);

  EXPECT_EQ(which, 2);
}

TEST(FactoryTest, PrintRegisteredListsKey) {
  auto &factory = quick_dot_ai::Factory::Instance();
  factory.registerModel("UT/Printable",
                        [](json &, json &, json &) { return nullptr; });

  std::ostringstream os;
  factory.printRegistered(os);
  EXPECT_NE(os.str().find("UT/Printable"), std::string::npos);
}

} // namespace
