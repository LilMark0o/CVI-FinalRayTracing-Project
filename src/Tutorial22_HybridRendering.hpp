/*
 *  Copyright 2019-2025 Diligent Graphics LLC
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  In no event and under no legal theory, whether in tort (including negligence),
 *  contract, or otherwise, unless required by applicable law (such as deliberate
 *  and grossly negligent acts) or agreed to in writing, shall any Contributor be
 *  liable for any damages, including any direct, indirect, special, incidental,
 *  or consequential damages of any character arising as a result of this License or
 *  out of the use or inability to use the software (including but not limited to damages
 *  for loss of goodwill, work stoppage, computer failure or malfunction, or any and
 *  all other commercial damages or losses), even if such Contributor has been advised
 *  of the possibility of such damages.
 */

#pragma once

#include "SampleBase.hpp"
#include "BasicMath.hpp"
#include "FirstPersonCamera.hpp"

namespace Diligent
{

    // We only need a 3x3 matrix, but in Vulkan and Metal, the rows of a float3x3 matrix are aligned to 16 bytes,
    // which is effectively a float4x3 matrix.
    // In DirectX, the rows of a float3x3 matrix are not aligned.
    // We will use a float4x3 for compatibility between all APIs.
    struct float4x3
    {
        float m00 = 0.f;
        float m01 = 0.f;
        float m02 = 0.f;
        float m03 = 0.f; // Unused

        float m10 = 0.f;
        float m11 = 0.f;
        float m12 = 0.f;
        float m13 = 0.f; // Unused

        float m20 = 0.f;
        float m21 = 0.f;
        float m22 = 0.f;
        float m23 = 0.f; // Unused

        float4x3() {}

        template <typename MatType>
        float4x3(const MatType &Other) : // clang-format off
        m00{Other.m00}, m01{Other.m01}, m02{Other.m02}, 
        m10{Other.m10}, m11{Other.m11}, m12{Other.m12}, 
        m20{Other.m20}, m21{Other.m21}, m22{Other.m22}
        // clang-format on
        {
        }
    };

    namespace HLSL
    {
#include "../assets/Structures.fxh"
    }

    enum class SpaceshipMovementType
    {
        ORBITAL = 0,      // Orbita alrededor de un punto
        LINEAR_PATROL,    // Patrulla en línea recta
        FORMATION_FOLLOW, // Sigue un patrón de formación
        RANDOM_WANDER,    // Movimiento aleatorio
        HOVER,            // Flotación con movimiento sutil
        SPIRAL,           // Movimiento en espiral
        FIGURE_EIGHT      // Movimiento en forma de ocho
    };

    struct SpaceshipDynamics
    {
        float3 basePosition;    // Posición base/centro de movimiento
        float3 currentVelocity; // Velocidad actual
        float3 targetVelocity;  // Velocidad objetivo
        float3 angularVelocity; // Velocidad angular (rotación)
        float3 formationOffset; // Offset para movimientos de formación

        SpaceshipMovementType movementType;

        // Parámetros específicos del movimiento
        float orbitRadius;     // Radio de órbita
        float orbitSpeed;      // Velocidad orbital
        float orbitPhase;      // Fase actual en la órbita
        float patrolDistance;  // Distancia de patrullaje
        float patrolSpeed;     // Velocidad de patrullaje
        float hoverRange;      // Rango de flotación
        float spiralRadius;    // Radio del espiral
        float spiralSpeed;     // Velocidad del espiral
        float figureEightSize; // Tamaño del patrón en ocho

        float timeOffset; // Offset temporal para variación
        float scale;      // Escala de la nave

        // Estado interno
        float patrolDirection; // Dirección actual de patrullaje (-1 o 1)
        float currentTime;     // Tiempo acumulado
    };

    class Tutorial22_HybridRendering final : public SampleBase
    {
    public:
        virtual void ModifyEngineInitInfo(const ModifyEngineInitInfoAttribs &Attribs) override final;
        virtual void Initialize(const SampleInitInfo &InitInfo) override final;

        virtual void Render() override final;
        virtual void Update(double CurrTime, double ElapsedTime, bool DoUpdateUI) override final;

        virtual const Char *GetSampleName() const override final { return "Tutorial22: Hybrid rendering"; }

        virtual void WindowResize(Uint32 Width, Uint32 Height) override final;

    protected:
        virtual void UpdateUI() override final;

    private:
        void CreateScene();
        void CreateSceneMaterials(uint2 &CubeMaterialRange, Uint32 &GroundMaterial, std::vector<HLSL::MaterialAttribs> &Materials);
        void CreateBuildingMaterials(uint2 &BuildingMaterialRange, std::vector<HLSL::MaterialAttribs> &Materials, Uint32 SamplerInd);
        void CreateSpaceshipMaterials(uint2 &SpaceshipMaterialRange, std::vector<HLSL::MaterialAttribs> &Materials, Uint32 SamplerInd);
        void CreateSceneObjects(const uint2 CubeMaterialRange, const uint2 BuildingMaterialRange, const uint2 SpaceshipMaterialRange, const Uint32 GroundMaterial);
        void RecreateSceneObjects();
        void CreateSceneAccelStructs();
        void UpdateTLAS();
        void CreateRasterizationPSO(IShaderSourceInputStreamFactory *pShaderSourceFactory);
        void CreatePostProcessPSO(IShaderSourceInputStreamFactory *pShaderSourceFactory);
        void CreateRayTracingPSO(IShaderSourceInputStreamFactory *pShaderSourceFactory);

        // Nueva función para actualizar movimiento de naves
        void UpdateSpaceshipMovement(float deltaTime);

        // Pipeline resource signature for scene resources used by the ray-tracing PSO
        RefCntAutoPtr<IPipelineResourceSignature> m_pRayTracingSceneResourcesSign;
        // Pipeline resource signature for screen resources used by the ray-tracing PSO
        RefCntAutoPtr<IPipelineResourceSignature> m_pRayTracingScreenResourcesSign;

        // Ray-tracing PSO
        RefCntAutoPtr<IPipelineState> m_RayTracingPSO;
        // Scene resources for ray-tracing PSO
        RefCntAutoPtr<IShaderResourceBinding> m_RayTracingSceneSRB;
        // Screen resources for ray-tracing PSO
        RefCntAutoPtr<IShaderResourceBinding> m_RayTracingScreenSRB;

        // G-buffer rendering PSO and SRB
        RefCntAutoPtr<IPipelineState> m_RasterizationPSO;
        RefCntAutoPtr<IShaderResourceBinding> m_RasterizationSRB;

        // Post-processing PSO and SRB
        RefCntAutoPtr<IPipelineState> m_PostProcessPSO;
        RefCntAutoPtr<IShaderResourceBinding> m_PostProcessSRB;

        // Simple implementation of a mesh
        struct Mesh
        {
            String Name;

            RefCntAutoPtr<IBottomLevelAS> BLAS;
            RefCntAutoPtr<IBuffer> VertexBuffer;
            RefCntAutoPtr<IBuffer> IndexBuffer;

            Uint32 NumVertices = 0;
            Uint32 NumIndices = 0;
            Uint32 FirstIndex = 0;  // Offset in the index buffer if IB and VB are shared between multiple meshes
            Uint32 FirstVertex = 0; // Offset in the vertex buffer
        };
        static Mesh CreateTexturedPlaneMesh(IRenderDevice *pDevice, float2 UVScale);
        static Mesh CreateTexturedBuildingMesh(IRenderDevice *pDevice, float2 UVScale, float3 Dimensions);
        static Mesh CreateSpaceshipMesh(IRenderDevice *pDevice, float2 UVScale);

        // Objects with the same mesh are grouped for instanced draw call
        struct InstancedObjects
        {
            Uint32 MeshInd = 0;             // Index in m_Scene.Meshes
            Uint32 ObjectAttribsOffset = 0; // Offset in m_Scene.ObjectAttribsBuffer
            Uint32 NumObjects = 0;          // Number of instances for a draw call
        };

        struct DynamicObject
        {
            Uint32 ObjectAttribsIndex = 0; // Index in m_Scene.ObjectAttribsBuffer
        };

        struct Scene
        {
            std::vector<InstancedObjects> ObjectInstances;
            std::vector<DynamicObject> DynamicObjects;
            std::vector<HLSL::ObjectAttribs> Objects; // CPU-visible array of HLSL::ObjectAttribs

            // Resources used by shaders
            std::vector<Mesh> Meshes;
            RefCntAutoPtr<IBuffer> MaterialAttribsBuffer;
            RefCntAutoPtr<IBuffer> ObjectAttribsBuffer; // GPU-visible array of HLSL::ObjectAttribs
            std::vector<RefCntAutoPtr<ITexture>> Textures;
            std::vector<RefCntAutoPtr<ISampler>> Samplers;
            RefCntAutoPtr<IBuffer> ObjectConstants;

            // Resources for ray tracing
            RefCntAutoPtr<ITopLevelAS> TLAS;
            RefCntAutoPtr<IBuffer> TLASInstancesBuffer; // Used to update TLAS
            RefCntAutoPtr<IBuffer> TLASScratchBuffer;   // Used to update TLAS

            // Scene state variables
            uint2 CubeMaterialRange;
            uint2 BuildingMaterialRange;
            uint2 SpaceshipMaterialRange;
            Uint32 GroundMaterial;
            Uint32 PlaneMeshId;
            Uint32 BuildingMeshId;
            Uint32 SpaceshipMeshId;

            // Clear scene objects but keep resources
            void ClearObjects()
            {
                ObjectInstances.clear();
                DynamicObjects.clear();
                Objects.clear();
            }
        };
        Scene m_Scene;

        // Constants shared between all PSOs
        RefCntAutoPtr<IBuffer> m_Constants;

        FirstPersonCamera m_Camera;

        struct GBuffer
        {
            RefCntAutoPtr<ITexture> Color;
            RefCntAutoPtr<ITexture> Normal;
            RefCntAutoPtr<ITexture> Depth;
        };

        const uint2 m_BlockSize = {8, 8};
        TEXTURE_FORMAT m_ColorTargetFormat = TEX_FORMAT_RGBA8_UNORM;
        TEXTURE_FORMAT m_NormalTargetFormat = TEX_FORMAT_RGBA16_FLOAT;
        TEXTURE_FORMAT m_DepthTargetFormat = TEX_FORMAT_D32_FLOAT;
        TEXTURE_FORMAT m_RayTracedTexFormat = TEX_FORMAT_RGBA16_FLOAT;

        GBuffer m_GBuffer;
        RefCntAutoPtr<ITexture> m_RayTracedTex;

        float3 m_LightDir = normalize(float3{-0.3f, -0.05f, 0.4f});

        int m_DrawMode = 0;

        // Building settings
        float m_DayNightFactor = 1.0f;  // 0.0 = noche, 1.0 = día (por defecto día)
        float m_BuildingDensity = 1.3f; // Controls building density multiplier (0.5-1.3)
        int m_BuildingCount = 100;      // Base number of buildings to create
        float3 m_DayLightDir = normalize(float3{-0.49f, -0.60f, 0.64f});
        float3 m_NightLightDir = normalize(float3{-0.3f, -0.05f, 0.4f});
        bool m_EditingDayLight = true; // True = editar luz diurna, False = editar luz nocturna

        int m_SpaceshipCount = 180; // Base number of spaceships to create

        // Scene settings
        bool m_NeedRecreateScene = false;

        // Spaceship dynamics
        std::vector<SpaceshipDynamics> m_SpaceshipDynamics;
        Uint32 m_SpaceshipStartIndex = 0; // Índice en m_Scene.Objects donde empiezan las naves

        // Movement control
        float m_SpaceshipSpeed = 1.0f;         // Multiplicador de velocidad global
        bool m_EnableSpaceshipMovement = true; // Control para activar/desactivar movimiento

        // Wave settings
        float m_WaveStrength = 0.15f; // Intensidad de las olas (0.0 a 1.0)
        float m_WaveSpeed = 2.0f;     // Velocidad de las olas (1.0 a 5.0)

        // Vulkan and DirectX require DXC shader compiler.
        // Metal uses the builtin glslang compiler.
#if PLATFORM_MACOS || PLATFORM_IOS || PLATFORM_TVOS
        const SHADER_COMPILER m_ShaderCompiler = SHADER_COMPILER_DEFAULT;
#else
        const SHADER_COMPILER m_ShaderCompiler = SHADER_COMPILER_DXC;
#endif
    };

} // namespace Diligent