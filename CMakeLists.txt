cmake_minimum_required (VERSION 3.10)

project(Tutorial22_HybridRendering CXX)

add_sample_app(Tutorial22_HybridRendering
    IDE_FOLDER
        DiligentSamples/Tutorials
    SOURCES
        src/Tutorial22_HybridRendering.cpp
    INCLUDES
        src/Tutorial22_HybridRendering.hpp
    SHADERS
        assets/Structures.fxh
        assets/Utils.fxh
        assets/RayQueryMtl.fxh
        assets/Rasterization.vsh
        assets/Rasterization.psh
        assets/PostProcess.vsh
        assets/PostProcess.psh
        assets/RayTracing.csh
    ASSETS
        assets/metal1.jpg
        assets/metal2.jpg
        assets/metal3.jpg
        assets/Marble.jpg
        assets/Building1.png
        assets/Building2.png
        assets/Building3.png
        assets/Water.png
        assets/Water.jpeg
    DXC_REQUIRED
        YES
)