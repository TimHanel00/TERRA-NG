target_sources(terra
        PRIVATE
        integrands.hpp)

add_subdirectory(operators)
add_subdirectory(quadrature)
add_subdirectory(sl)
add_subdirectory(shell)