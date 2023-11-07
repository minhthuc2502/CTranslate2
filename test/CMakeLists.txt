if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cxxopts/include")
  message(FATAL_ERROR "The client dependency repository (cxxopts) is missing! "
    "You probably didn't clone the project with --recursive. You can include it "
    "by calling \"git submodule update --init --recursive\"")
endif()

add_executable(CTranslate2Server
  main.cpp
  )

add_executable(TranslatorTest
  translate_test.cpp
)

#[[add_executable(GeneratorTest
  generate_test.cpp
)]]

target_include_directories(CTranslate2Server
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cxxopts/include
  )
target_link_libraries(CTranslate2Server
  PRIVATE ${PROJECT_NAME}
  sentencepiece
  boost_program_options
)

target_link_libraries(TranslatorTest
  PRIVATE ${PROJECT_NAME}
  sentencepiece
  boost_program_options
)

#[[
target_link_libraries(GeneratorTest
  PRIVATE ${PROJECT_NAME}
  sentencepiece
  boost_program_options
)
]]

set_target_properties(CTranslate2Server PROPERTIES OUTPUT_NAME ct2-CTranslate2Server)
set_target_properties(TranslatorTest PROPERTIES OUTPUT_NAME ct2-TranslatorTest)
#set_target_properties(GeneratorTest PROPERTIES OUTPUT_NAME ct2-GeneratorTest)

install(
  TARGETS CTranslate2Server
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
