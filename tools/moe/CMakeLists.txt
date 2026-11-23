set(TARGET llama-moe-profile)
add_executable(${TARGET} profile_hot_experts.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TARGET} RUNTIME)

# Offline LRU / reuse-distance replay of a routing trace (no llama dependency:
# it only reads the *-routes.jsonl that llama-moe-profile produced).
set(TARGET llama-moe-locality)
add_executable(${TARGET} locality_profile.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TARGET} RUNTIME)

# Per-layer LRU load curve at 8..256 slots (feeds tools/moe/solve_k.py).
set(TARGET llama-moe-window-sweep)
add_executable(${TARGET} window_sweep.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TARGET} RUNTIME)
