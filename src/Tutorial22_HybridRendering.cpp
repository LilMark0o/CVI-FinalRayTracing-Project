#include "Tutorial22_HybridRendering.hpp"

#include "MapHelper.hpp"
#include "GraphicsUtilities.h"
#include "TextureUtilities.h"
#include "ShaderMacroHelper.hpp"
#include "imgui.h"
#include "ImGuiUtils.hpp"
#include "../imGuIZMO.quat/imGuIZMO.h"
#include "Align.hpp"

namespace Diligent
{

    static_assert(sizeof(HLSL::GlobalConstants) % 16 == 0, "Structure must be 16-byte aligned");
    static_assert(sizeof(HLSL::ObjectConstants) % 16 == 0, "Structure must be 16-byte aligned");

    SampleBase *CreateSample()
    {
        return new Tutorial22_HybridRendering();
    }

    void Tutorial22_HybridRendering::CreateSceneMaterials(uint2 &CubeMaterialRange, Uint32 &GroundMaterial, std::vector<HLSL::MaterialAttribs> &Materials)
    {
        Uint32 AnisotropicClampSampInd = 0;
        Uint32 AnisotropicWrapSampInd = 0;

        // Create samplers
        {
            const SamplerDesc AnisotropicClampSampler{
                FILTER_TYPE_ANISOTROPIC, FILTER_TYPE_ANISOTROPIC, FILTER_TYPE_ANISOTROPIC,
                TEXTURE_ADDRESS_CLAMP, TEXTURE_ADDRESS_CLAMP, TEXTURE_ADDRESS_CLAMP, 0.f, 8 //
            };
            const SamplerDesc AnisotropicWrapSampler{
                FILTER_TYPE_ANISOTROPIC, FILTER_TYPE_ANISOTROPIC, FILTER_TYPE_ANISOTROPIC,
                TEXTURE_ADDRESS_WRAP, TEXTURE_ADDRESS_WRAP, TEXTURE_ADDRESS_WRAP, 0.f, 8 //
            };

            RefCntAutoPtr<ISampler> pSampler;
            m_pDevice->CreateSampler(AnisotropicClampSampler, &pSampler);
            AnisotropicClampSampInd = static_cast<Uint32>(m_Scene.Samplers.size());
            m_Scene.Samplers.push_back(std::move(pSampler));

            pSampler = nullptr;
            m_pDevice->CreateSampler(AnisotropicWrapSampler, &pSampler);
            AnisotropicWrapSampInd = static_cast<Uint32>(m_Scene.Samplers.size());
            m_Scene.Samplers.push_back(std::move(pSampler));
        }

        const auto LoadMaterial = [&](const char *ColorMapName, const float4 &BaseColor, Uint32 SamplerInd) //
        {
            TextureLoadInfo loadInfo;
            loadInfo.IsSRGB = true;
            loadInfo.GenerateMips = true;
            RefCntAutoPtr<ITexture> Tex;
            CreateTextureFromFile(ColorMapName, loadInfo, m_pDevice, &Tex);
            VERIFY_EXPR(Tex);

            HLSL::MaterialAttribs mtr;
            mtr.SampInd = SamplerInd;
            mtr.BaseColorMask = BaseColor;
            mtr.BaseColorTexInd = static_cast<Uint32>(m_Scene.Textures.size());
            m_Scene.Textures.push_back(std::move(Tex));
            Materials.push_back(mtr);
        };

        // Cube materials
        CubeMaterialRange.x = static_cast<Uint32>(Materials.size());
        LoadMaterial("DGLogo0.png", float4{1.f}, AnisotropicClampSampInd);
        LoadMaterial("DGLogo1.png", float4{1.f}, AnisotropicClampSampInd);
        LoadMaterial("DGLogo2.png", float4{1.f}, AnisotropicClampSampInd);
        LoadMaterial("DGLogo3.png", float4{1.f}, AnisotropicClampSampInd);
        CubeMaterialRange.y = static_cast<Uint32>(Materials.size());

        // Ground material
        GroundMaterial = static_cast<Uint32>(Materials.size());
        LoadMaterial("Water.jpeg", float4{0.6f, 0.8f, 1.0f, 0.8f}, AnisotropicWrapSampInd); // Blue tint for water
    }

    void Tutorial22_HybridRendering::CreateBuildingMaterials(uint2 &BuildingMaterialRange, std::vector<HLSL::MaterialAttribs> &Materials, Uint32 SamplerInd)
    {
        BuildingMaterialRange.x = static_cast<Uint32>(Materials.size());

        // Define the LoadMaterial function here with the same functionality as in CreateSceneMaterials
        const auto LoadMaterial = [&](const char *ColorMapName, const float4 &BaseColor, Uint32 SamplerInd) //
        {
            TextureLoadInfo loadInfo;
            loadInfo.IsSRGB = true;
            loadInfo.GenerateMips = true;
            RefCntAutoPtr<ITexture> Tex;
            CreateTextureFromFile(ColorMapName, loadInfo, m_pDevice, &Tex);
            VERIFY_EXPR(Tex);

            HLSL::MaterialAttribs mtr;
            mtr.SampInd = SamplerInd;
            mtr.BaseColorMask = BaseColor;
            mtr.BaseColorTexInd = static_cast<Uint32>(m_Scene.Textures.size());
            m_Scene.Textures.push_back(std::move(Tex));
            Materials.push_back(mtr);
        };

        // Now use standard textures since you may not have actual building textures
        LoadMaterial("Building1.png", float4{0.8f, 0.8f, 1.0f, 1.0f}, SamplerInd); // Bluish tint
        LoadMaterial("Building2.png", float4{0.9f, 0.9f, 0.9f, 1.0f}, SamplerInd); // Grayish tint
        LoadMaterial("Building3.png", float4{1.0f, 0.9f, 0.8f, 1.0f}, SamplerInd); // Yellowish tint

        BuildingMaterialRange.y = static_cast<Uint32>(Materials.size());
    }

    Tutorial22_HybridRendering::Mesh Tutorial22_HybridRendering::CreateTexturedPlaneMesh(IRenderDevice *pDevice, float2 UVScale)
    {
        Mesh PlaneMesh;
        PlaneMesh.Name = "Ground";

        {
            struct PlaneVertex // Alias for HLSL::Vertex
            {
                float3 pos;
                float3 norm;
                float2 uv;
            };
            static_assert(sizeof(PlaneVertex) == sizeof(HLSL::Vertex), "Vertex size mismatch");

            // clang-format off
        const PlaneVertex Vertices[] = 
        {
            {float3{-1, 0, -1}, float3{0, 1, 0}, float2{0,         0        }},
            {float3{ 1, 0, -1}, float3{0, 1, 0}, float2{UVScale.x, 0        }},
            {float3{-1, 0,  1}, float3{0, 1, 0}, float2{0,         UVScale.y}},
            {float3{ 1, 0,  1}, float3{0, 1, 0}, float2{UVScale.x, UVScale.y}}
        };
            // clang-format on
            PlaneMesh.NumVertices = _countof(Vertices);

            BufferDesc VBDesc;
            VBDesc.Name = "Plane vertex buffer";
            VBDesc.Usage = USAGE_IMMUTABLE;
            VBDesc.BindFlags = BIND_VERTEX_BUFFER | BIND_SHADER_RESOURCE | BIND_RAY_TRACING;
            VBDesc.Size = sizeof(Vertices);
            VBDesc.Mode = BUFFER_MODE_STRUCTURED;
            VBDesc.ElementByteStride = sizeof(Vertices[0]);
            BufferData VBData{Vertices, VBDesc.Size};
            pDevice->CreateBuffer(VBDesc, &VBData, &PlaneMesh.VertexBuffer);
        }

        {
            const Uint32 Indices[] = {0, 2, 3, 3, 1, 0};
            PlaneMesh.NumIndices = _countof(Indices);

            BufferDesc IBDesc;
            IBDesc.Name = "Plane index buffer";
            IBDesc.BindFlags = BIND_INDEX_BUFFER | BIND_SHADER_RESOURCE | BIND_RAY_TRACING;
            IBDesc.Size = sizeof(Indices);
            IBDesc.Mode = BUFFER_MODE_STRUCTURED;
            IBDesc.ElementByteStride = sizeof(Indices[0]);
            BufferData IBData{Indices, IBDesc.Size};
            pDevice->CreateBuffer(IBDesc, &IBData, &PlaneMesh.IndexBuffer);
        }

        return PlaneMesh;
    }

    Tutorial22_HybridRendering::Mesh Tutorial22_HybridRendering::CreateTexturedBuildingMesh(IRenderDevice *pDevice, float2 UVScale, float3 Dimensions)
    {
        Mesh BuildingMesh;
        BuildingMesh.Name = "Building";

        {
            // Define the building vertex structure with position, normal, and UV coordinates
            struct BuildingVertex
            {
                float3 pos;
                float3 norm;
                float2 uv;
            };
            static_assert(sizeof(BuildingVertex) == sizeof(HLSL::Vertex), "Vertex size mismatch");

            // We'll create vertices for each face separately to ensure correct normals and UVs
            // This means we'll have 24 vertices (6 faces × 4 vertices per face)
            BuildingVertex Vertices[24];

            // Compute half-dimensions for convenience
            float3 halfDim = Dimensions * 0.5f;

            // Front face (z = -d/2)
            Vertices[0] = {float3{-halfDim.x, -halfDim.y, -halfDim.z}, float3{0, 0, -1}, float2{0, 0}};
            Vertices[1] = {float3{-halfDim.x, halfDim.y, -halfDim.z}, float3{0, 0, -1}, float2{0, UVScale.y}};
            Vertices[2] = {float3{halfDim.x, halfDim.y, -halfDim.z}, float3{0, 0, -1}, float2{UVScale.x, UVScale.y}};
            Vertices[3] = {float3{halfDim.x, -halfDim.y, -halfDim.z}, float3{0, 0, -1}, float2{UVScale.x, 0}};

            // Back face (z = d/2)
            Vertices[4] = {float3{halfDim.x, -halfDim.y, halfDim.z}, float3{0, 0, 1}, float2{0, 0}};
            Vertices[5] = {float3{halfDim.x, halfDim.y, halfDim.z}, float3{0, 0, 1}, float2{0, UVScale.y}};
            Vertices[6] = {float3{-halfDim.x, halfDim.y, halfDim.z}, float3{0, 0, 1}, float2{UVScale.x, UVScale.y}};
            Vertices[7] = {float3{-halfDim.x, -halfDim.y, halfDim.z}, float3{0, 0, 1}, float2{UVScale.x, 0}};

            // Right face (x = w/2)
            Vertices[8] = {float3{halfDim.x, -halfDim.y, -halfDim.z}, float3{1, 0, 0}, float2{0, 0}};
            Vertices[9] = {float3{halfDim.x, halfDim.y, -halfDim.z}, float3{1, 0, 0}, float2{0, UVScale.y}};
            Vertices[10] = {float3{halfDim.x, halfDim.y, halfDim.z}, float3{1, 0, 0}, float2{UVScale.x, UVScale.y}};
            Vertices[11] = {float3{halfDim.x, -halfDim.y, halfDim.z}, float3{1, 0, 0}, float2{UVScale.x, 0}};

            // Left face (x = -w/2)
            Vertices[12] = {float3{-halfDim.x, -halfDim.y, halfDim.z}, float3{-1, 0, 0}, float2{0, 0}};
            Vertices[13] = {float3{-halfDim.x, halfDim.y, halfDim.z}, float3{-1, 0, 0}, float2{0, UVScale.y}};
            Vertices[14] = {float3{-halfDim.x, halfDim.y, -halfDim.z}, float3{-1, 0, 0}, float2{UVScale.x, UVScale.y}};
            Vertices[15] = {float3{-halfDim.x, -halfDim.y, -halfDim.z}, float3{-1, 0, 0}, float2{UVScale.x, 0}};

            // Top face (y = h/2)
            Vertices[16] = {float3{-halfDim.x, halfDim.y, -halfDim.z}, float3{0, 1, 0}, float2{0, 0}};
            Vertices[17] = {float3{-halfDim.x, halfDim.y, halfDim.z}, float3{0, 1, 0}, float2{0, UVScale.y}};
            Vertices[18] = {float3{halfDim.x, halfDim.y, halfDim.z}, float3{0, 1, 0}, float2{UVScale.x, UVScale.y}};
            Vertices[19] = {float3{halfDim.x, halfDim.y, -halfDim.z}, float3{0, 1, 0}, float2{UVScale.x, 0}};

            // Bottom face (y = -h/2)
            Vertices[20] = {float3{-halfDim.x, -halfDim.y, halfDim.z}, float3{0, -1, 0}, float2{0, 0}};
            Vertices[21] = {float3{-halfDim.x, -halfDim.y, -halfDim.z}, float3{0, -1, 0}, float2{0, UVScale.y}};
            Vertices[22] = {float3{halfDim.x, -halfDim.y, -halfDim.z}, float3{0, -1, 0}, float2{UVScale.x, UVScale.y}};
            Vertices[23] = {float3{halfDim.x, -halfDim.y, halfDim.z}, float3{0, -1, 0}, float2{UVScale.x, 0}};

            BuildingMesh.NumVertices = _countof(Vertices);

            BufferDesc VBDesc;
            VBDesc.Name = "Building vertex buffer";
            VBDesc.Usage = USAGE_IMMUTABLE;
            VBDesc.BindFlags = BIND_VERTEX_BUFFER | BIND_SHADER_RESOURCE | BIND_RAY_TRACING;
            VBDesc.Size = sizeof(Vertices);
            VBDesc.Mode = BUFFER_MODE_STRUCTURED;
            VBDesc.ElementByteStride = sizeof(Vertices[0]);
            BufferData VBData{Vertices, VBDesc.Size};
            pDevice->CreateBuffer(VBDesc, &VBData, &BuildingMesh.VertexBuffer);
        }

        {
            // Define indices for the 6 faces (2 triangles per face)
            // Each face now uses its own set of 4 vertices with proper normals and UVs
            const Uint32 Indices[] =
                {
                    // Front face (CCW winding)
                    0, 1, 2,
                    0, 2, 3,

                    // Back face (CCW winding)
                    4, 5, 6,
                    4, 6, 7,

                    // Right face (CCW winding)
                    8, 9, 10,
                    8, 10, 11,

                    // Left face (CCW winding)
                    12, 13, 14,
                    12, 14, 15,

                    // Top face (CCW winding)
                    16, 17, 18,
                    16, 18, 19,

                    // Bottom face (CCW winding)
                    20, 21, 22,
                    20, 22, 23};

            BuildingMesh.NumIndices = _countof(Indices);

            BufferDesc IBDesc;
            IBDesc.Name = "Building index buffer";
            IBDesc.BindFlags = BIND_INDEX_BUFFER | BIND_SHADER_RESOURCE | BIND_RAY_TRACING;
            IBDesc.Size = sizeof(Indices);
            IBDesc.Mode = BUFFER_MODE_STRUCTURED;
            IBDesc.ElementByteStride = sizeof(Indices[0]);
            BufferData IBData{Indices, IBDesc.Size};
            pDevice->CreateBuffer(IBDesc, &IBData, &BuildingMesh.IndexBuffer);
        }

        return BuildingMesh;
    }

    void Tutorial22_HybridRendering::CreateSceneObjects(const uint2 CubeMaterialRange, const uint2 BuildingMaterialRange, const Uint32 GroundMaterial)
    {
        // Save material ranges for recreating objects later
        m_Scene.CubeMaterialRange = CubeMaterialRange;
        m_Scene.BuildingMaterialRange = BuildingMaterialRange;
        m_Scene.GroundMaterial = GroundMaterial;

        // Clear any existing objects from the scene
        m_Scene.ClearObjects();

        Uint32 PlaneMeshId = 0;
        Uint32 BuildingMeshId = 0;

        // Create meshes if they don't exist yet
        if (m_Scene.Meshes.empty())
        {
            // Create plane mesh
            Mesh PlaneMesh = CreateTexturedPlaneMesh(m_pDevice, float2{25});

            // Create building mesh with better proportions for high-rise buildings
            // Using a smaller building size than before
            Mesh BuildingMesh = CreateTexturedBuildingMesh(m_pDevice, float2{1}, float3{2.0f, 8.0f, 2.0f});

            const RayTracingProperties &RTProps = m_pDevice->GetAdapterInfo().RayTracing;

            // Plane mesh will be first in the buffer
            PlaneMesh.FirstVertex = 0;
            PlaneMesh.FirstIndex = 0;

            // Building mesh data will reside after the plane. Offsets must be properly aligned!
            BuildingMesh.FirstVertex = AlignUp(PlaneMesh.NumVertices * Uint32{sizeof(HLSL::Vertex)},
                                               RTProps.VertexBufferAlignment) /
                                       sizeof(HLSL::Vertex);
            BuildingMesh.FirstIndex = AlignUp(PlaneMesh.NumIndices * Uint32{sizeof(uint)},
                                              RTProps.IndexBufferAlignment) /
                                      sizeof(uint);

            // Create merged vertex buffer
            {
                BufferDesc VBDesc;
                VBDesc.Name = "Shared vertex buffer";
                VBDesc.BindFlags = BIND_VERTEX_BUFFER | BIND_SHADER_RESOURCE | BIND_RAY_TRACING;
                VBDesc.Size = (Uint64{BuildingMesh.FirstVertex} + Uint64{BuildingMesh.NumVertices}) * sizeof(HLSL::Vertex);
                VBDesc.Mode = BUFFER_MODE_STRUCTURED;
                VBDesc.ElementByteStride = sizeof(HLSL::Vertex);

                RefCntAutoPtr<IBuffer> pSharedVB;
                m_pDevice->CreateBuffer(VBDesc, nullptr, &pSharedVB);

                // Copy plane vertices
                m_pImmediateContext->CopyBuffer(PlaneMesh.VertexBuffer, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                                                pSharedVB, PlaneMesh.FirstVertex * sizeof(HLSL::Vertex),
                                                PlaneMesh.NumVertices * sizeof(HLSL::Vertex),
                                                RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

                // Copy building vertices
                m_pImmediateContext->CopyBuffer(BuildingMesh.VertexBuffer, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                                                pSharedVB, BuildingMesh.FirstVertex * sizeof(HLSL::Vertex),
                                                BuildingMesh.NumVertices * sizeof(HLSL::Vertex),
                                                RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

                PlaneMesh.VertexBuffer = pSharedVB;
                BuildingMesh.VertexBuffer = pSharedVB;
            }

            // Create merged index buffer
            {
                BufferDesc IBDesc;
                IBDesc.Name = "Shared index buffer";
                IBDesc.BindFlags = BIND_INDEX_BUFFER | BIND_SHADER_RESOURCE | BIND_RAY_TRACING;
                IBDesc.Size = (Uint64{BuildingMesh.FirstIndex} + Uint64{BuildingMesh.NumIndices}) * sizeof(uint);
                IBDesc.Mode = BUFFER_MODE_STRUCTURED;
                IBDesc.ElementByteStride = sizeof(uint);

                RefCntAutoPtr<IBuffer> pSharedIB;
                m_pDevice->CreateBuffer(IBDesc, nullptr, &pSharedIB);

                // Copy plane indices
                m_pImmediateContext->CopyBuffer(PlaneMesh.IndexBuffer, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                                                pSharedIB, PlaneMesh.FirstIndex * sizeof(uint),
                                                PlaneMesh.NumIndices * sizeof(uint),
                                                RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

                // Copy building indices
                m_pImmediateContext->CopyBuffer(BuildingMesh.IndexBuffer, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                                                pSharedIB, BuildingMesh.FirstIndex * sizeof(uint),
                                                BuildingMesh.NumIndices * sizeof(uint),
                                                RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

                PlaneMesh.IndexBuffer = pSharedIB;
                BuildingMesh.IndexBuffer = pSharedIB;
            }

            PlaneMeshId = static_cast<Uint32>(m_Scene.Meshes.size());
            m_Scene.Meshes.push_back(PlaneMesh);
            BuildingMeshId = static_cast<Uint32>(m_Scene.Meshes.size());
            m_Scene.Meshes.push_back(BuildingMesh);

            // Store mesh IDs for later use
            m_Scene.PlaneMeshId = PlaneMeshId;
            m_Scene.BuildingMeshId = BuildingMeshId;
        }
        else
        {
            // Reuse existing meshes
            PlaneMeshId = m_Scene.PlaneMeshId;
            BuildingMeshId = m_Scene.BuildingMeshId;
        }

        // Calculate building positions with density factor
        const int TotalBuildingCount = static_cast<int>(m_BuildingCount * m_BuildingDensity);

        // Make the area larger to fit more buildings
        const float AreaSize = 25.0f + (m_BuildingDensity * 10.0f);

        // Seed for random positioning
        std::srand(42); // Fixed seed for repeatable results

        // Track building positions and dimensions to avoid overlaps
        struct BuildingPosition
        {
            float x, z;   // Center position
            float radius; // Building's radius (half-width)
        };

        std::vector<BuildingPosition> placedBuildings;

        // Reduced minimum distance between buildings for denser placement
        const float MinDistance = 2.0f / m_BuildingDensity;

        // Create buildings with non-overlapping positions
        int attemptsRemaining = TotalBuildingCount * 10; // Limit total placement attempts
        int buildingsPlaced = 0;

        while (buildingsPlaced < TotalBuildingCount && attemptsRemaining > 0)
        {
            attemptsRemaining--;

            // Generate random position within area
            float X = (std::rand() % 200) / 100.0f * AreaSize - AreaSize;
            float Z = (std::rand() % 200) / 100.0f * AreaSize - AreaSize;

            // Keep away from center (where camera starts)
            if (std::abs(X) < 5.0f && std::abs(Z) < 5.0f)
            {
                X += (X < 0) ? -5.0f : 5.0f;
                Z += (Z < 0) ? -5.0f : 5.0f;
            }

            // Randomize building properties - using smaller scale range for more uniform buildings
            float Angle = (std::rand() % 100) / 100.0f * PI_F;
            float Y = (std::rand() % 100) / 100.0f * 0.5f; // Height off ground (reduced)

            // Use smaller scale range for buildings to fit more of them
            float Scale = 0.4f + (std::rand() % 100) / 100.0f * 0.3f;

            // Calculate building footprint radius - smaller than before
            float buildingRadius = Scale * 1.0f;

            // Check for overlaps with existing buildings
            bool overlaps = false;
            for (const auto &building : placedBuildings)
            {
                float dx = X - building.x;
                float dz = Z - building.z;
                float distance = std::sqrt(dx * dx + dz * dz);

                // If distance is less than sum of radii plus minimum spacing, we have an overlap
                if (distance < (buildingRadius + building.radius + MinDistance))
                {
                    overlaps = true;
                    break;
                }
            }

            // Skip this position if it overlaps
            if (overlaps)
            {
                continue;
            }

            // Position buildings closer to the water by using a lower Y offset
            // The water level is at -1.5f, so we'll position buildings just above that
            const float4x4 ModelMat = float4x4::RotationY(Angle * PI_F) * float4x4::Scale(Scale) * float4x4::Translation(X * 2.0f, Y * 1.0f - 1.0f, Z * 2.0f);

            HLSL::ObjectAttribs obj;
            obj.ModelMat = ModelMat.Transpose();
            obj.NormalMat = obj.ModelMat;
            obj.MaterialId = (buildingsPlaced % (BuildingMaterialRange.y - BuildingMaterialRange.x)) + BuildingMaterialRange.x;
            obj.MeshId = BuildingMeshId;
            obj.FirstIndex = m_Scene.Meshes[obj.MeshId].FirstIndex;
            obj.FirstVertex = m_Scene.Meshes[obj.MeshId].FirstVertex;
            m_Scene.Objects.push_back(obj);

            // Add this building to our placed list
            placedBuildings.push_back({X, Z, buildingRadius});
            buildingsPlaced++;
        }

        // Add building instance object
        InstancedObjects InstObj;
        InstObj.ObjectAttribsOffset = 0;
        InstObj.MeshInd = BuildingMeshId;
        InstObj.NumObjects = static_cast<Uint32>(m_Scene.Objects.size());
        m_Scene.ObjectInstances.push_back(InstObj);

        // Create ground plane object (water)
        InstObj.ObjectAttribsOffset = static_cast<Uint32>(m_Scene.Objects.size());
        InstObj.MeshInd = PlaneMeshId;
        {
            HLSL::ObjectAttribs obj;
            // Position the water level lower and make it larger to match expanded area
            obj.ModelMat = (float4x4::Scale(80.f, 1.f, 80.f) * float4x4::Translation(0.f, -1.5f, 0.f)).Transpose();
            obj.NormalMat = float3x3::Identity();
            obj.MaterialId = GroundMaterial;
            obj.MeshId = PlaneMeshId;
            obj.FirstIndex = m_Scene.Meshes[obj.MeshId].FirstIndex;
            obj.FirstVertex = m_Scene.Meshes[obj.MeshId].FirstVertex;
            m_Scene.Objects.push_back(obj);
        }
        InstObj.NumObjects = static_cast<Uint32>(m_Scene.Objects.size()) - InstObj.ObjectAttribsOffset;
        m_Scene.ObjectInstances.push_back(InstObj);
    }

    void Tutorial22_HybridRendering::CreateSceneAccelStructs()
    {
        // Validate that we have meshes
        if (m_Scene.Meshes.empty())
            return;

        // Create and build bottom-level acceleration structure for each mesh
        {
            RefCntAutoPtr<IBuffer> pScratchBuffer;

            for (Mesh &mesh : m_Scene.Meshes)
            {
                // Release existing BLAS if any
                mesh.BLAS.Release();

                // Create BLAS
                BLASTriangleDesc Triangles;
                {
                    Triangles.GeometryName = mesh.Name.c_str();
                    Triangles.MaxVertexCount = mesh.NumVertices;
                    Triangles.VertexValueType = VT_FLOAT32;
                    Triangles.VertexComponentCount = 3;
                    Triangles.MaxPrimitiveCount = mesh.NumIndices / 3;
                    Triangles.IndexType = VT_UINT32;

                    const std::string BLASName{mesh.Name + " BLAS"};

                    BottomLevelASDesc ASDesc;
                    ASDesc.Name = BLASName.c_str();
                    ASDesc.Flags = RAYTRACING_BUILD_AS_PREFER_FAST_TRACE;
                    ASDesc.pTriangles = &Triangles;
                    ASDesc.TriangleCount = 1;
                    m_pDevice->CreateBLAS(ASDesc, &mesh.BLAS);
                }

                // Create or reuse scratch buffer; this will insert the barrier between BuildBLAS invocations, which may be suboptimal.
                if (!pScratchBuffer || pScratchBuffer->GetDesc().Size < mesh.BLAS->GetScratchBufferSizes().Build)
                {
                    BufferDesc BuffDesc;
                    BuffDesc.Name = "BLAS Scratch Buffer";
                    BuffDesc.Usage = USAGE_DEFAULT;
                    BuffDesc.BindFlags = BIND_RAY_TRACING;
                    BuffDesc.Size = mesh.BLAS->GetScratchBufferSizes().Build;

                    pScratchBuffer = nullptr;
                    m_pDevice->CreateBuffer(BuffDesc, nullptr, &pScratchBuffer);
                }

                // Build BLAS
                BLASBuildTriangleData TriangleData;
                TriangleData.GeometryName = Triangles.GeometryName;
                TriangleData.pVertexBuffer = mesh.VertexBuffer;
                TriangleData.VertexStride = mesh.VertexBuffer->GetDesc().ElementByteStride;
                TriangleData.VertexOffset = Uint64{mesh.FirstVertex} * Uint64{TriangleData.VertexStride};
                TriangleData.VertexCount = mesh.NumVertices;
                TriangleData.VertexValueType = Triangles.VertexValueType;
                TriangleData.VertexComponentCount = Triangles.VertexComponentCount;
                TriangleData.pIndexBuffer = mesh.IndexBuffer;
                TriangleData.IndexOffset = Uint64{mesh.FirstIndex} * Uint64{mesh.IndexBuffer->GetDesc().ElementByteStride};
                TriangleData.PrimitiveCount = Triangles.MaxPrimitiveCount;
                TriangleData.IndexType = Triangles.IndexType;
                TriangleData.Flags = RAYTRACING_GEOMETRY_FLAG_OPAQUE;

                BuildBLASAttribs Attribs;
                Attribs.pBLAS = mesh.BLAS;
                Attribs.pTriangleData = &TriangleData;
                Attribs.TriangleDataCount = 1;

                // Scratch buffer will be used to store temporary data during the BLAS build.
                // Previous content in the scratch buffer will be discarded.
                Attribs.pScratchBuffer = pScratchBuffer;

                // Allow engine to change resource states.
                Attribs.BLASTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
                Attribs.GeometryTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
                Attribs.ScratchBufferTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;

                m_pImmediateContext->BuildBLAS(Attribs);
            }
        }

        // Create new TLAS
        {
            m_Scene.TLAS.Release(); // Ensure we're not leaking resources

            TopLevelASDesc TLASDesc;
            TLASDesc.Name = "Scene TLAS";
            TLASDesc.MaxInstanceCount = static_cast<Uint32>(m_Scene.Objects.size());
            TLASDesc.Flags = RAYTRACING_BUILD_AS_PREFER_FAST_TRACE; // Don't use UPDATE flag
            m_pDevice->CreateTLAS(TLASDesc, &m_Scene.TLAS);
        }
    }

    void Tutorial22_HybridRendering::UpdateTLAS()
    {
        const Uint32 NumInstances = static_cast<Uint32>(m_Scene.Objects.size());

        // Create scratch buffer
        if (!m_Scene.TLASScratchBuffer ||
            m_Scene.TLASScratchBuffer->GetDesc().Size < m_Scene.TLAS->GetScratchBufferSizes().Build)
        {
            BufferDesc BuffDesc;
            BuffDesc.Name = "TLAS Scratch Buffer";
            BuffDesc.Usage = USAGE_DEFAULT;
            BuffDesc.BindFlags = BIND_RAY_TRACING;
            BuffDesc.Size = std::max(m_Scene.TLAS->GetScratchBufferSizes().Build, m_Scene.TLAS->GetScratchBufferSizes().Update);
            m_pDevice->CreateBuffer(BuffDesc, nullptr, &m_Scene.TLASScratchBuffer);
        }

        // Create instance buffer
        if (!m_Scene.TLASInstancesBuffer ||
            m_Scene.TLASInstancesBuffer->GetDesc().Size < Uint64{TLAS_INSTANCE_DATA_SIZE} * Uint64{NumInstances})
        {
            BufferDesc BuffDesc;
            BuffDesc.Name = "TLAS Instance Buffer";
            BuffDesc.Usage = USAGE_DEFAULT;
            BuffDesc.BindFlags = BIND_RAY_TRACING;
            BuffDesc.Size = Uint64{TLAS_INSTANCE_DATA_SIZE} * Uint64{NumInstances};
            m_pDevice->CreateBuffer(BuffDesc, nullptr, &m_Scene.TLASInstancesBuffer);
        }

        // Setup instances
        std::vector<TLASBuildInstanceData> Instances(NumInstances);
        std::vector<String> InstanceNames(NumInstances);
        for (Uint32 i = 0; i < NumInstances; ++i)
        {
            const HLSL::ObjectAttribs &Obj = m_Scene.Objects[i];
            TLASBuildInstanceData &Inst = Instances[i];
            std::string &Name = InstanceNames[i];
            const Mesh &mesh = m_Scene.Meshes[Obj.MeshId];
            const float4x4 ModelMat = Obj.ModelMat.Transpose();

            Name = mesh.Name + " Instance (" + std::to_string(i) + ")";

            Inst.InstanceName = Name.c_str();
            Inst.pBLAS = mesh.BLAS;
            Inst.Mask = 0xFF;

            // CustomId will be read in shader by RayQuery::CommittedInstanceID()
            Inst.CustomId = i;

            Inst.Transform.SetRotation(ModelMat.Data(), 4);
            Inst.Transform.SetTranslation(ModelMat.m30, ModelMat.m31, ModelMat.m32);
        }

        // Build TLAS - always do a full build, never update
        BuildTLASAttribs Attribs;
        Attribs.pTLAS = m_Scene.TLAS;
        Attribs.Update = false; // Never update, always rebuild

        // Scratch buffer will be used to store temporary data during TLAS build or update.
        // Previous content in the scratch buffer will be discarded.
        Attribs.pScratchBuffer = m_Scene.TLASScratchBuffer;

        // Instance buffer will store instance data during TLAS build or update.
        // Previous content in the instance buffer will be discarded.
        Attribs.pInstanceBuffer = m_Scene.TLASInstancesBuffer;

        // Instances will be converted to the format that is required by the graphics driver and copied to the instance buffer.
        Attribs.pInstances = Instances.data();
        Attribs.InstanceCount = NumInstances;

        // Allow engine to change resource states.
        Attribs.TLASTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
        Attribs.BLASTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
        Attribs.InstanceBufferTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
        Attribs.ScratchBufferTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;

        m_pImmediateContext->BuildTLAS(Attribs);
    }

    void Tutorial22_HybridRendering::RecreateSceneObjects()
    {
        // First, save the current material ranges and ground material
        const auto CubeMaterialRange = m_Scene.CubeMaterialRange;
        const auto BuildingMaterialRange = m_Scene.BuildingMaterialRange;
        const auto GroundMaterial = m_Scene.GroundMaterial;

        // Fully release all existing bindings and resources that will be recreated
        m_RasterizationSRB.Release();
        m_RayTracingSceneSRB.Release();

        // Release all acceleration structures
        m_Scene.TLAS.Release();
        m_Scene.TLASInstancesBuffer.Release();
        m_Scene.TLASScratchBuffer.Release();

        // Release object buffers that will be recreated
        m_Scene.ObjectAttribsBuffer.Release();

        // Clear existing objects but preserve meshes
        m_Scene.ClearObjects();

        // Recreate objects from scratch
        CreateSceneObjects(CubeMaterialRange, BuildingMaterialRange, GroundMaterial);

        // Create buffer for object attribs - we need to do this here explicitly
        {
            BufferDesc BuffDesc;
            BuffDesc.Name = "Object attribs buffer";
            BuffDesc.Usage = USAGE_DEFAULT;
            BuffDesc.BindFlags = BIND_SHADER_RESOURCE;
            BuffDesc.Size = static_cast<Uint64>(sizeof(m_Scene.Objects[0]) * m_Scene.Objects.size());
            BuffDesc.Mode = BUFFER_MODE_STRUCTURED;
            BuffDesc.ElementByteStride = sizeof(m_Scene.Objects[0]);
            m_pDevice->CreateBuffer(BuffDesc, nullptr, &m_Scene.ObjectAttribsBuffer);
        }

        // Create new acceleration structures
        CreateSceneAccelStructs();

        // Create new SRBs with delay-initialized resources
        m_RasterizationPSO->CreateShaderResourceBinding(&m_RasterizationSRB, true);
        m_pRayTracingSceneResourcesSign->CreateShaderResourceBinding(&m_RayTracingSceneSRB, true);

        // Initialize all bindings for rasterization SRB
        m_RasterizationSRB->GetVariableByName(SHADER_TYPE_VERTEX, "g_Constants")->Set(m_Constants);
        m_RasterizationSRB->GetVariableByName(SHADER_TYPE_VERTEX, "g_ObjectConst")->Set(m_Scene.ObjectConstants);
        m_RasterizationSRB->GetVariableByName(SHADER_TYPE_VERTEX, "g_ObjectAttribs")->Set(m_Scene.ObjectAttribsBuffer->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
        m_RasterizationSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_MaterialAttribs")->Set(m_Scene.MaterialAttribsBuffer->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));

        // Bind textures
        {
            const Uint32 NumTextures = static_cast<Uint32>(m_Scene.Textures.size());
            std::vector<IDeviceObject *> ppTextures(NumTextures);
            for (Uint32 i = 0; i < NumTextures; ++i)
                ppTextures[i] = m_Scene.Textures[i]->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
            m_RasterizationSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Textures")->SetArray(ppTextures.data(), 0, NumTextures);
        }

        // Bind samplers
        {
            const Uint32 NumSamplers = static_cast<Uint32>(m_Scene.Samplers.size());
            std::vector<IDeviceObject *> ppSamplers(NumSamplers);
            for (Uint32 i = 0; i < NumSamplers; ++i)
                ppSamplers[i] = m_Scene.Samplers[i];
            m_RasterizationSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Samplers")->SetArray(ppSamplers.data(), 0, NumSamplers);
        }

        // Set bindings for ray tracing scene resources
        m_RayTracingSceneSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "g_TLAS")->Set(m_Scene.TLAS);
        m_RayTracingSceneSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "g_Constants")->Set(m_Constants);
        m_RayTracingSceneSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "g_ObjectAttribs")->Set(m_Scene.ObjectAttribsBuffer->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
        m_RayTracingSceneSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "g_MaterialAttribs")->Set(m_Scene.MaterialAttribsBuffer->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));

        // Bind mesh geometry buffers
        m_RayTracingSceneSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "g_VertexBuffer")->Set(m_Scene.Meshes[0].VertexBuffer->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
        m_RayTracingSceneSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "g_IndexBuffer")->Set(m_Scene.Meshes[0].IndexBuffer->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));

        // Bind material textures
        {
            const Uint32 NumTextures = static_cast<Uint32>(m_Scene.Textures.size());
            std::vector<IDeviceObject *> ppTextures(NumTextures);
            for (Uint32 i = 0; i < NumTextures; ++i)
                ppTextures[i] = m_Scene.Textures[i]->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
            m_RayTracingSceneSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "g_Textures")->SetArray(ppTextures.data(), 0, NumTextures);
        }

        // Bind samplers
        {
            const Uint32 NumSamplers = static_cast<Uint32>(m_Scene.Samplers.size());
            std::vector<IDeviceObject *> ppSamplers(NumSamplers);
            for (Uint32 i = 0; i < NumSamplers; ++i)
                ppSamplers[i] = m_Scene.Samplers[i];
            m_RayTracingSceneSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "g_Samplers")->SetArray(ppSamplers.data(), 0, NumSamplers);
        }
    }

    void Tutorial22_HybridRendering::CreateScene()
    {
        uint2 CubeMaterialRange;
        uint2 BuildingMaterialRange;
        Uint32 GroundMaterial;
        std::vector<HLSL::MaterialAttribs> Materials;
        CreateSceneMaterials(CubeMaterialRange, GroundMaterial, Materials);
        CreateBuildingMaterials(BuildingMaterialRange, Materials, 0); // Use sampler index 0
        CreateSceneObjects(CubeMaterialRange, BuildingMaterialRange, GroundMaterial);
        CreateSceneAccelStructs();

        // Create buffer for object attribs
        m_Scene.ObjectAttribsBuffer.Release(); // Make sure to release before creating a new one
        {
            BufferDesc BuffDesc;
            BuffDesc.Name = "Object attribs buffer";
            BuffDesc.Usage = USAGE_DEFAULT;
            BuffDesc.BindFlags = BIND_SHADER_RESOURCE;
            BuffDesc.Size = static_cast<Uint64>(sizeof(m_Scene.Objects[0]) * m_Scene.Objects.size());
            BuffDesc.Mode = BUFFER_MODE_STRUCTURED;
            BuffDesc.ElementByteStride = sizeof(m_Scene.Objects[0]);
            m_pDevice->CreateBuffer(BuffDesc, nullptr, &m_Scene.ObjectAttribsBuffer);
        }

        // Create and initialize buffer for material attribs
        m_Scene.MaterialAttribsBuffer.Release(); // Make sure to release before creating a new one
        {
            BufferDesc BuffDesc;
            BuffDesc.Name = "Material attribs buffer";
            BuffDesc.Usage = USAGE_DEFAULT;
            BuffDesc.BindFlags = BIND_SHADER_RESOURCE;
            BuffDesc.Size = static_cast<Uint64>(sizeof(Materials[0]) * Materials.size());
            BuffDesc.Mode = BUFFER_MODE_STRUCTURED;
            BuffDesc.ElementByteStride = sizeof(Materials[0]);

            BufferData BuffData{Materials.data(), BuffDesc.Size};
            m_pDevice->CreateBuffer(BuffDesc, &BuffData, &m_Scene.MaterialAttribsBuffer);
        }

        // Create dynamic buffer for scene object constants (unique for each draw call)
        m_Scene.ObjectConstants.Release(); // Make sure to release before creating a new one
        {
            BufferDesc BuffDesc;
            BuffDesc.Name = "Global constants buffer";
            BuffDesc.Usage = USAGE_DYNAMIC;
            BuffDesc.BindFlags = BIND_UNIFORM_BUFFER;
            BuffDesc.Size = sizeof(HLSL::ObjectConstants);
            BuffDesc.CPUAccessFlags = CPU_ACCESS_WRITE;
            m_pDevice->CreateBuffer(BuffDesc, nullptr, &m_Scene.ObjectConstants);
        }
    }

    void Tutorial22_HybridRendering::CreateRasterizationPSO(IShaderSourceInputStreamFactory *pShaderSourceFactory)
    {
        // Create PSO for rendering to GBuffer

        ShaderMacroHelper Macros;
        Macros.AddShaderMacro("NUM_TEXTURES", static_cast<Uint32>(m_Scene.Textures.size()));
        Macros.AddShaderMacro("NUM_SAMPLERS", static_cast<Uint32>(m_Scene.Samplers.size()));

        GraphicsPipelineStateCreateInfo PSOCreateInfo;

        PSOCreateInfo.PSODesc.Name = "Rasterization PSO";
        PSOCreateInfo.PSODesc.PipelineType = PIPELINE_TYPE_GRAPHICS;

        PSOCreateInfo.GraphicsPipeline.NumRenderTargets = 2;
        PSOCreateInfo.GraphicsPipeline.RTVFormats[0] = m_ColorTargetFormat;
        PSOCreateInfo.GraphicsPipeline.RTVFormats[1] = m_NormalTargetFormat;
        PSOCreateInfo.GraphicsPipeline.DSVFormat = m_DepthTargetFormat;
        PSOCreateInfo.GraphicsPipeline.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        PSOCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode = CULL_MODE_BACK;
        PSOCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthEnable = True;

        ShaderCreateInfo ShaderCI;
        ShaderCI.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
        ShaderCI.ShaderCompiler = m_ShaderCompiler;
        ShaderCI.pShaderSourceStreamFactory = pShaderSourceFactory;
        ShaderCI.Macros = Macros;

        RefCntAutoPtr<IShader> pVS;
        {
            ShaderCI.Desc.ShaderType = SHADER_TYPE_VERTEX;
            ShaderCI.EntryPoint = "main";
            ShaderCI.Desc.Name = "Rasterization VS";
            ShaderCI.FilePath = "Rasterization.vsh";
            m_pDevice->CreateShader(ShaderCI, &pVS);
        }

        RefCntAutoPtr<IShader> pPS;
        {
            ShaderCI.Desc.ShaderType = SHADER_TYPE_PIXEL;
            ShaderCI.EntryPoint = "main";
            ShaderCI.Desc.Name = "Rasterization PS";
            ShaderCI.FilePath = "Rasterization.psh";
            m_pDevice->CreateShader(ShaderCI, &pPS);
        }

        PSOCreateInfo.pVS = pVS;
        PSOCreateInfo.pPS = pPS;

        LayoutElement LayoutElems[] =
            {
                LayoutElement{0, 0, 3, VT_FLOAT32, False},
                LayoutElement{1, 0, 3, VT_FLOAT32, False},
                LayoutElement{2, 0, 2, VT_FLOAT32, False} //
            };
        PSOCreateInfo.GraphicsPipeline.InputLayout.LayoutElements = LayoutElems;
        PSOCreateInfo.GraphicsPipeline.InputLayout.NumElements = _countof(LayoutElems);

        PSOCreateInfo.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE;
        PSOCreateInfo.PSODesc.ResourceLayout.DefaultVariableMergeStages = SHADER_TYPE_VERTEX | SHADER_TYPE_PIXEL;

        m_pDevice->CreateGraphicsPipelineState(PSOCreateInfo, &m_RasterizationPSO);

        m_RasterizationPSO->CreateShaderResourceBinding(&m_RasterizationSRB);
        m_RasterizationSRB->GetVariableByName(SHADER_TYPE_VERTEX, "g_Constants")->Set(m_Constants);
        m_RasterizationSRB->GetVariableByName(SHADER_TYPE_VERTEX, "g_ObjectConst")->Set(m_Scene.ObjectConstants);
        m_RasterizationSRB->GetVariableByName(SHADER_TYPE_VERTEX, "g_ObjectAttribs")->Set(m_Scene.ObjectAttribsBuffer->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
        m_RasterizationSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_MaterialAttribs")->Set(m_Scene.MaterialAttribsBuffer->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));

        // Bind textures
        {
            const Uint32 NumTextures = static_cast<Uint32>(m_Scene.Textures.size());
            std::vector<IDeviceObject *> ppTextures(NumTextures);
            for (Uint32 i = 0; i < NumTextures; ++i)
                ppTextures[i] = m_Scene.Textures[i]->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
            m_RasterizationSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Textures")->SetArray(ppTextures.data(), 0, NumTextures);
        }

        // Bind samplers
        {
            const Uint32 NumSamplers = static_cast<Uint32>(m_Scene.Samplers.size());
            std::vector<IDeviceObject *> ppSamplers(NumSamplers);
            for (Uint32 i = 0; i < NumSamplers; ++i)
                ppSamplers[i] = m_Scene.Samplers[i];
            m_RasterizationSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Samplers")->SetArray(ppSamplers.data(), 0, NumSamplers);
        }
    }

    void Tutorial22_HybridRendering::CreatePostProcessPSO(IShaderSourceInputStreamFactory *pShaderSourceFactory)
    {
        // Create PSO for post process pass

        GraphicsPipelineStateCreateInfo PSOCreateInfo;

        PSOCreateInfo.PSODesc.Name = "Post process PSO";
        PSOCreateInfo.PSODesc.PipelineType = PIPELINE_TYPE_GRAPHICS;

        PSOCreateInfo.GraphicsPipeline.NumRenderTargets = 1;
        PSOCreateInfo.GraphicsPipeline.RTVFormats[0] = m_pSwapChain->GetDesc().ColorBufferFormat;
        PSOCreateInfo.GraphicsPipeline.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        PSOCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthEnable = false;
        PSOCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = false;

        PSOCreateInfo.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE;

        ShaderCreateInfo ShaderCI;
        ShaderCI.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
        ShaderCI.ShaderCompiler = m_ShaderCompiler;
        ShaderCI.pShaderSourceStreamFactory = pShaderSourceFactory;

        RefCntAutoPtr<IShader> pVS;
        {
            ShaderCI.Desc.ShaderType = SHADER_TYPE_VERTEX;
            ShaderCI.EntryPoint = "main";
            ShaderCI.Desc.Name = "Post process VS";
            ShaderCI.FilePath = "PostProcess.vsh";
            m_pDevice->CreateShader(ShaderCI, &pVS);
        }

        RefCntAutoPtr<IShader> pPS;
        {
            ShaderCI.Desc.ShaderType = SHADER_TYPE_PIXEL;
            ShaderCI.EntryPoint = "main";
            ShaderCI.Desc.Name = "Post process PS";
            ShaderCI.FilePath = "PostProcess.psh";
            m_pDevice->CreateShader(ShaderCI, &pPS);
        }

        PSOCreateInfo.pVS = pVS;
        PSOCreateInfo.pPS = pPS;

        m_pDevice->CreateGraphicsPipelineState(PSOCreateInfo, &m_PostProcessPSO);
    }

    void Tutorial22_HybridRendering::CreateRayTracingPSO(IShaderSourceInputStreamFactory *pShaderSourceFactory)
    {
        // Create compute shader that performs inline ray tracing

        ShaderMacroHelper Macros;
        Macros.AddShaderMacro("NUM_TEXTURES", static_cast<Uint32>(m_Scene.Textures.size()));
        Macros.AddShaderMacro("NUM_SAMPLERS", static_cast<Uint32>(m_Scene.Samplers.size()));

        ComputePipelineStateCreateInfo PSOCreateInfo;

        PSOCreateInfo.PSODesc.PipelineType = PIPELINE_TYPE_COMPUTE;

        const Uint32 NumTextures = static_cast<Uint32>(m_Scene.Textures.size());
        const Uint32 NumSamplers = static_cast<Uint32>(m_Scene.Samplers.size());

        // Split the resources of the ray tracing PSO into two groups.
        // The first group will contain scene resources. These resources
        // may be bound only once.
        // The second group will contain screen-dependent resources.
        // These resources will need to be bound every time the screen is resized.

        // Resource signature for scene resources
        {
            PipelineResourceSignatureDesc PRSDesc;
            PRSDesc.Name = "Ray tracing scene resources";

            // clang-format off
        const PipelineResourceDesc Resources[] =
        {
            {SHADER_TYPE_COMPUTE, "g_TLAS",            1,           SHADER_RESOURCE_TYPE_ACCEL_STRUCT},
            {SHADER_TYPE_COMPUTE, "g_Constants",       1,           SHADER_RESOURCE_TYPE_CONSTANT_BUFFER},
            {SHADER_TYPE_COMPUTE, "g_ObjectAttribs",   1,           SHADER_RESOURCE_TYPE_BUFFER_SRV},
            {SHADER_TYPE_COMPUTE, "g_MaterialAttribs", 1,           SHADER_RESOURCE_TYPE_BUFFER_SRV},
            {SHADER_TYPE_COMPUTE, "g_VertexBuffer",    1,           SHADER_RESOURCE_TYPE_BUFFER_SRV},
            {SHADER_TYPE_COMPUTE, "g_IndexBuffer",     1,           SHADER_RESOURCE_TYPE_BUFFER_SRV},
            {SHADER_TYPE_COMPUTE, "g_Textures",        NumTextures, SHADER_RESOURCE_TYPE_TEXTURE_SRV},
            {SHADER_TYPE_COMPUTE, "g_Samplers",        NumSamplers, SHADER_RESOURCE_TYPE_SAMPLER}
        };
            // clang-format on
            PRSDesc.BindingIndex = 0;
            PRSDesc.Resources = Resources;
            PRSDesc.NumResources = _countof(Resources);
            m_pDevice->CreatePipelineResourceSignature(PRSDesc, &m_pRayTracingSceneResourcesSign);
            VERIFY_EXPR(m_pRayTracingSceneResourcesSign);
        }

        // Resource signature for screen resources
        {
            PipelineResourceSignatureDesc PRSDesc;
            PRSDesc.Name = "Ray tracing screen resources";

            // clang-format off
        const PipelineResourceDesc Resources[] =
        {
            {SHADER_TYPE_COMPUTE, "g_RayTracedTex",   1, SHADER_RESOURCE_TYPE_TEXTURE_UAV},
            {SHADER_TYPE_COMPUTE, "g_GBuffer_Normal", 1, SHADER_RESOURCE_TYPE_TEXTURE_SRV},
            {SHADER_TYPE_COMPUTE, "g_GBuffer_Depth",  1, SHADER_RESOURCE_TYPE_TEXTURE_SRV}
        };
            // clang-format on
            PRSDesc.BindingIndex = 1;
            PRSDesc.Resources = Resources;
            PRSDesc.NumResources = _countof(Resources);
            m_pDevice->CreatePipelineResourceSignature(PRSDesc, &m_pRayTracingScreenResourcesSign);
            VERIFY_EXPR(m_pRayTracingScreenResourcesSign);
        }

        IPipelineResourceSignature *ppSignatures[]{m_pRayTracingSceneResourcesSign, m_pRayTracingScreenResourcesSign};
        PSOCreateInfo.ppResourceSignatures = ppSignatures;
        PSOCreateInfo.ResourceSignaturesCount = _countof(ppSignatures);

        ShaderCreateInfo ShaderCI;
        ShaderCI.Desc.ShaderType = SHADER_TYPE_COMPUTE;
        ShaderCI.pShaderSourceStreamFactory = pShaderSourceFactory;
        ShaderCI.EntryPoint = "CSMain";
        ShaderCI.Macros = Macros;

        if (m_pDevice->GetDeviceInfo().IsMetalDevice())
        {
            // HLSL and MSL are very similar, so we can use the same code for all
            // platforms, with some macros help.
            ShaderCI.ShaderCompiler = SHADER_COMPILER_DEFAULT;
            ShaderCI.SourceLanguage = SHADER_SOURCE_LANGUAGE_MSL;
        }
        else
        {
            // Inline ray tracing requires shader model 6.5
            // Only DXC can compile HLSL for ray tracing.
            ShaderCI.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
            ShaderCI.ShaderCompiler = SHADER_COMPILER_DXC;
            ShaderCI.HLSLVersion = {6, 5};
        }

        ShaderCI.Desc.Name = "Ray tracing CS";
        ShaderCI.FilePath = "RayTracing.csh";
        if (m_pDevice->GetDeviceInfo().IsMetalDevice())
        {
            // The shader uses macros that are not supported by MSL parser in Metal backend
            ShaderCI.CompileFlags = SHADER_COMPILE_FLAG_SKIP_REFLECTION;
        }
        RefCntAutoPtr<IShader> pCS;
        m_pDevice->CreateShader(ShaderCI, &pCS);
        PSOCreateInfo.pCS = pCS;

        PSOCreateInfo.PSODesc.Name = "Ray tracing PSO";
        m_pDevice->CreateComputePipelineState(PSOCreateInfo, &m_RayTracingPSO);
        VERIFY_EXPR(m_RayTracingPSO);

        // Initialize SRB containing scene resources
        m_pRayTracingSceneResourcesSign->CreateShaderResourceBinding(&m_RayTracingSceneSRB);
        m_RayTracingSceneSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "g_TLAS")->Set(m_Scene.TLAS);
        m_RayTracingSceneSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "g_Constants")->Set(m_Constants);
        m_RayTracingSceneSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "g_ObjectAttribs")->Set(m_Scene.ObjectAttribsBuffer->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
        m_RayTracingSceneSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "g_MaterialAttribs")->Set(m_Scene.MaterialAttribsBuffer->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));

        // Bind mesh geometry buffers. All meshes use shared vertex and index buffers.
        m_RayTracingSceneSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "g_VertexBuffer")->Set(m_Scene.Meshes[0].VertexBuffer->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
        m_RayTracingSceneSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "g_IndexBuffer")->Set(m_Scene.Meshes[0].IndexBuffer->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));

        // Bind material textures
        {
            std::vector<IDeviceObject *> ppTextures(NumTextures);
            for (Uint32 i = 0; i < NumTextures; ++i)
                ppTextures[i] = m_Scene.Textures[i]->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
            m_RayTracingSceneSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "g_Textures")->SetArray(ppTextures.data(), 0, NumTextures);
        }

        // Bind samplers
        {
            std::vector<IDeviceObject *> ppSamplers(NumSamplers);
            for (Uint32 i = 0; i < NumSamplers; ++i)
                ppSamplers[i] = m_Scene.Samplers[i];
            m_RayTracingSceneSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "g_Samplers")->SetArray(ppSamplers.data(), 0, NumSamplers);
        }
    }

    void Tutorial22_HybridRendering::Initialize(const SampleInitInfo &InitInfo)
    {
        SampleBase::Initialize(InitInfo);

        // RayTracing feature indicates that some of ray tracing functionality is supported.
        // Acceleration structures are always supported if RayTracing feature is enabled.
        // Inline ray tracing may be unsupported by old DirectX 12 drivers or if this feature is not supported by Vulkan.
        if ((m_pDevice->GetAdapterInfo().RayTracing.CapFlags & RAY_TRACING_CAP_FLAG_INLINE_RAY_TRACING) == 0)
        {
            UNSUPPORTED("Inline ray tracing is not supported by device");
            return;
        }

        // Setup camera.
        m_Camera.SetPos(float3{-15.7f, 3.7f, -5.8f});
        m_Camera.SetRotation(17.7f, -0.1f);
        m_Camera.SetRotationSpeed(0.005f);
        m_Camera.SetMoveSpeed(5.f);
        m_Camera.SetSpeedUpScales(5.f, 10.f);

        // Create buffer for constants that is shared between all PSOs
        {
            BufferDesc BuffDesc;
            BuffDesc.Name = "Global constants buffer";
            BuffDesc.BindFlags = BIND_UNIFORM_BUFFER;
            BuffDesc.Size = sizeof(HLSL::GlobalConstants);
            m_pDevice->CreateBuffer(BuffDesc, nullptr, &m_Constants);
        }

        RefCntAutoPtr<IShaderSourceInputStreamFactory> pShaderSourceFactory;
        m_pEngineFactory->CreateDefaultShaderSourceStreamFactory(nullptr, &pShaderSourceFactory);

        CreateScene();
        CreateRasterizationPSO(pShaderSourceFactory);
        CreatePostProcessPSO(pShaderSourceFactory);
        CreateRayTracingPSO(pShaderSourceFactory);
    }

    void Tutorial22_HybridRendering::ModifyEngineInitInfo(const ModifyEngineInitInfoAttribs &Attribs)
    {
        SampleBase::ModifyEngineInitInfo(Attribs);

        // Require ray tracing feature.
        Attribs.EngineCI.Features.RayTracing = DEVICE_FEATURE_STATE_ENABLED;
    }

    void Tutorial22_HybridRendering::Render()
    {
        // Update constants
        {
            const float4x4 ViewProj = m_Camera.GetViewMatrix() * m_Camera.GetProjMatrix();

            HLSL::GlobalConstants GConst;
            GConst.ViewProj = ViewProj.Transpose();
            GConst.ViewProjInv = ViewProj.Inverse().Transpose();
            GConst.LightDir = normalize(-m_LightDir);
            GConst.CameraPos = float4(m_Camera.GetPos(), 0.f);
            GConst.DrawMode = m_DrawMode;
            GConst.MaxRayLength = 100.f;
            GConst.AmbientLight = 0.1f;
            m_pImmediateContext->UpdateBuffer(m_Constants, 0, static_cast<Uint32>(sizeof(GConst)), &GConst, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

            // Update transformation for scene objects
            m_pImmediateContext->UpdateBuffer(m_Scene.ObjectAttribsBuffer, 0, static_cast<Uint32>(sizeof(HLSL::ObjectAttribs) * m_Scene.Objects.size()),
                                              m_Scene.Objects.data(), RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        }

        UpdateTLAS();

        // Rasterization pass
        {
            ITextureView *RTVs[] = //
                {
                    m_GBuffer.Color->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET),
                    m_GBuffer.Normal->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET) //
                };
            ITextureView *pDSV = m_GBuffer.Depth->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL);
            m_pImmediateContext->SetRenderTargets(_countof(RTVs), RTVs, pDSV, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

            // All transitions for render targets happened in SetRenderTargets()
            const float ClearColor[4] = {};
            m_pImmediateContext->ClearRenderTarget(RTVs[0], ClearColor, RESOURCE_STATE_TRANSITION_MODE_NONE);
            m_pImmediateContext->ClearRenderTarget(RTVs[1], ClearColor, RESOURCE_STATE_TRANSITION_MODE_NONE);
            m_pImmediateContext->ClearDepthStencil(pDSV, CLEAR_DEPTH_FLAG, 1.f, 0, RESOURCE_STATE_TRANSITION_MODE_NONE);

            m_pImmediateContext->SetPipelineState(m_RasterizationPSO);
            m_pImmediateContext->CommitShaderResources(m_RasterizationSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

            for (InstancedObjects &ObjInst : m_Scene.ObjectInstances)
            {
                Mesh &mesh = m_Scene.Meshes[ObjInst.MeshInd];
                IBuffer *VBs[] = {mesh.VertexBuffer};
                const Uint64 Offsets[] = {mesh.FirstVertex * sizeof(HLSL::Vertex)};

                m_pImmediateContext->SetVertexBuffers(0, _countof(VBs), VBs, Offsets, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
                m_pImmediateContext->SetIndexBuffer(mesh.IndexBuffer, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

                {
                    MapHelper<HLSL::ObjectConstants> ObjConstants{m_pImmediateContext, m_Scene.ObjectConstants, MAP_WRITE, MAP_FLAG_DISCARD};
                    ObjConstants->ObjectAttribsOffset = ObjInst.ObjectAttribsOffset;
                }

                DrawIndexedAttribs drawAttribs;
                drawAttribs.NumIndices = mesh.NumIndices;
                drawAttribs.NumInstances = ObjInst.NumObjects;
                drawAttribs.FirstIndexLocation = mesh.FirstIndex;
                drawAttribs.IndexType = VT_UINT32;
                drawAttribs.Flags = DRAW_FLAG_VERIFY_ALL;
                m_pImmediateContext->DrawIndexed(drawAttribs);
            }
        }

        // Ray tracing pass
        {
            DispatchComputeAttribs dispatchAttribs;
            dispatchAttribs.MtlThreadGroupSizeX = m_BlockSize.x;
            dispatchAttribs.MtlThreadGroupSizeY = m_BlockSize.y;
            dispatchAttribs.MtlThreadGroupSizeZ = 1;

            const TextureDesc &TexDesc = m_GBuffer.Color->GetDesc();
            dispatchAttribs.ThreadGroupCountX = (TexDesc.Width / m_BlockSize.x);
            dispatchAttribs.ThreadGroupCountY = (TexDesc.Height / m_BlockSize.y);

            m_pImmediateContext->SetPipelineState(m_RayTracingPSO);
            m_pImmediateContext->CommitShaderResources(m_RayTracingSceneSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
            m_pImmediateContext->CommitShaderResources(m_RayTracingScreenSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
            m_pImmediateContext->DispatchCompute(dispatchAttribs);
        }

        // Post process pass
        {
            ITextureView *pRTV = m_pSwapChain->GetCurrentBackBufferRTV();
            const float ClearColor[4] = {};
            m_pImmediateContext->SetRenderTargets(1, &pRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
            m_pImmediateContext->ClearRenderTarget(pRTV, ClearColor, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

            m_pImmediateContext->SetPipelineState(m_PostProcessPSO);
            m_pImmediateContext->CommitShaderResources(m_PostProcessSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

            m_pImmediateContext->SetVertexBuffers(0, 0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE, SET_VERTEX_BUFFERS_FLAG_RESET);
            m_pImmediateContext->SetIndexBuffer(nullptr, 0, RESOURCE_STATE_TRANSITION_MODE_NONE);

            m_pImmediateContext->Draw(DrawAttribs{3, DRAW_FLAG_VERIFY_ALL});
        }
    }

    void Tutorial22_HybridRendering::Update(double CurrTime, double ElapsedTime, bool DoUpdateUI)
    {
        SampleBase::Update(CurrTime, ElapsedTime, DoUpdateUI);

        const float dt = static_cast<float>(ElapsedTime);

        m_Camera.Update(m_InputController, dt);

        // Restrict camera movement
        float3 Pos = m_Camera.GetPos();
        const float3 MinXYZ{-75.f, 0.1f, -75.f};
        const float3 MaxXYZ{+75.f, +6.f, 75.f};
        if (Pos.x < MinXYZ.x || Pos.y < MinXYZ.y || Pos.z < MinXYZ.z ||
            Pos.x > MaxXYZ.x || Pos.y > MaxXYZ.y || Pos.z > MaxXYZ.z)
        {
            Pos = clamp(Pos, MinXYZ, MaxXYZ);
            m_Camera.SetPos(Pos);
            m_Camera.Update(m_InputController, 0);
        }

        // Recreate objects if density changed
        if (m_NeedRecreateScene)
        {
            RecreateSceneObjects();
            m_NeedRecreateScene = false;
        }

        // Buildings are now static, so we don't need to update them
    }

    void Tutorial22_HybridRendering::WindowResize(Uint32 Width, Uint32 Height)
    {
        if (Width == 0 || Height == 0)
            return;

        // Round to multiple of m_BlockSize
        Width = AlignUp(Width, m_BlockSize.x);
        Height = AlignUp(Height, m_BlockSize.y);

        // Update projection matrix.
        float AspectRatio = static_cast<float>(Width) / static_cast<float>(Height);
        m_Camera.SetProjAttribs(0.1f, 100.f, AspectRatio, PI_F / 4.f,
                                m_pSwapChain->GetDesc().PreTransform, m_pDevice->GetDeviceInfo().NDC.MinZ == -1);

        // Check if the image needs to be recreated.
        if (m_GBuffer.Color != nullptr &&
            m_GBuffer.Color->GetDesc().Width == Width &&
            m_GBuffer.Color->GetDesc().Height == Height)
            return;

        m_GBuffer = {};

        // Create window-size G-buffer textures.
        TextureDesc RTDesc;
        RTDesc.Name = "GBuffer Color";
        RTDesc.Type = RESOURCE_DIM_TEX_2D;
        RTDesc.Width = Width;
        RTDesc.Height = Height;
        RTDesc.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
        RTDesc.Format = m_ColorTargetFormat;
        m_pDevice->CreateTexture(RTDesc, nullptr, &m_GBuffer.Color);

        RTDesc.Name = "GBuffer Normal";
        RTDesc.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
        RTDesc.Format = m_NormalTargetFormat;
        m_pDevice->CreateTexture(RTDesc, nullptr, &m_GBuffer.Normal);

        RTDesc.Name = "GBuffer Depth";
        RTDesc.BindFlags = BIND_DEPTH_STENCIL | BIND_SHADER_RESOURCE;
        RTDesc.Format = m_DepthTargetFormat;
        m_pDevice->CreateTexture(RTDesc, nullptr, &m_GBuffer.Depth);

        RTDesc.Name = "Ray traced shadow & reflection";
        RTDesc.BindFlags = BIND_UNORDERED_ACCESS | BIND_SHADER_RESOURCE;
        RTDesc.Format = m_RayTracedTexFormat;
        m_RayTracedTex.Release();
        m_pDevice->CreateTexture(RTDesc, nullptr, &m_RayTracedTex);

        // Release old SRBs
        m_PostProcessSRB.Release();
        m_RayTracingScreenSRB.Release();

        // Create post-processing SRB
        m_PostProcessPSO->CreateShaderResourceBinding(&m_PostProcessSRB);
        m_PostProcessSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Constants")->Set(m_Constants);
        m_PostProcessSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_GBuffer_Color")->Set(m_GBuffer.Color->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
        m_PostProcessSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_GBuffer_Normal")->Set(m_GBuffer.Normal->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
        m_PostProcessSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_GBuffer_Depth")->Set(m_GBuffer.Depth->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
        m_PostProcessSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_RayTracedTex")->Set(m_RayTracedTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));

        // Create ray-tracing screen SRB
        m_pRayTracingScreenResourcesSign->CreateShaderResourceBinding(&m_RayTracingScreenSRB);
        m_RayTracingScreenSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "g_RayTracedTex")->Set(m_RayTracedTex->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS));
        m_RayTracingScreenSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "g_GBuffer_Depth")->Set(m_GBuffer.Depth->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
        m_RayTracingScreenSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "g_GBuffer_Normal")->Set(m_GBuffer.Normal->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
    }

    void Tutorial22_HybridRendering::UpdateUI()
    {
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        {
            ImGui::Combo("Render mode", &m_DrawMode,
                         "Shaded\0"
                         "G-buffer color\0"
                         "G-buffer normal\0"
                         "Diffuse lighting\0"
                         "Reflections\0"
                         "Fresnel term\0\0");

            // Building Density slider with tooltip
            {
                ImGui::Separator();
                ImGui::TextDisabled("Building Density");
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Controla qué tan densamente se colocan los edificios");

                float PrevBuildingDensity = m_BuildingDensity;
                ImGui::SliderFloat("##BuildingDensity", &m_BuildingDensity, 0.5f, 1.3f, "%.2f");

                // If building density changed, mark for recreation
                if (PrevBuildingDensity != m_BuildingDensity)
                {
                    m_NeedRecreateScene = true;
                }
            }

            // Building Count slider with tooltip
            {
                ImGui::TextDisabled("Building Count");
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Controla el número total de edificios");

                int PrevBuildingCount = m_BuildingCount;
                int BuildingCountInt = m_BuildingCount;
                ImGui::SliderInt("##BuildingCount", &BuildingCountInt, 20, 500);
                m_BuildingCount = BuildingCountInt;

                // If building count changed, mark for recreation
                if (PrevBuildingCount != m_BuildingCount)
                {
                    m_NeedRecreateScene = true;
                }
            }

            ImGui::Separator();

            // Light direction control
            ImGui::TextDisabled("Light Direction");
            if (ImGui::gizmo3D("##LightDirection", m_LightDir))
            {
                if (m_LightDir.y > -0.06f)
                {
                    m_LightDir.y = -0.06f;
                    m_LightDir = normalize(m_LightDir);
                }
            }
        }
        ImGui::End();
    }

} // namespace Diligent