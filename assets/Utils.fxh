
// Simple procedural sky and sun
float4 GetSkyColor(float3 Dir, float3 LightDir, float DayNightFactor)
{
    Dir.y += 0.075;
    Dir = normalize(Dir);

    float CosTheta = dot(Dir, LightDir);
    float ScatteringScale = pow(saturate(0.5 * (1.0 - CosTheta)), 0.2);    

    // Intensidad del sol basada en ciclo día/noche
    float SunIntensity = lerp(60.0, 50.0, DayNightFactor);
    float SunFactor = lerp(0.5, 5.0, DayNightFactor);
    float3 SkyColor = pow(saturate(CosTheta - 0.02), SunIntensity) * saturate(LightDir.y * SunFactor);
    
    // Interpolar entre colores de noche y día
    float3 DaySkyBase = float3(0.07, 0.11, 0.23); // Color base día
    float3 NightSkyBase = float3(0.004, 0.006, 0.03); // Color base noche
    float3 SkyBaseColor = lerp(NightSkyBase, DaySkyBase, DayNightFactor);
    
    float3 SkyDome = 
        SkyBaseColor *
        lerp(max(ScatteringScale, 0.1), 1.0, saturate(Dir.y)) / max(Dir.y, 0.01);
        
    // Ajustar intensidad del cielo
    float SkyDomeFactor = lerp(5.0, 13.0, DayNightFactor);
    SkyDome *= SkyDomeFactor / max(length(SkyDome), SkyDomeFactor);
    
    float3 Horizon = pow(SkyDome, float3(1.0, 1.0, 1.0) - SkyDome);
    SkyColor += lerp(Horizon, SkyDome / (SkyDome + 0.5), saturate(Dir.y * 2.0));
    
    // Ajustar intensidad general
    float ScatteringFactor = lerp(2.0, 10.0, DayNightFactor);
    SkyColor *= 1.0 + pow(1.0 - ScatteringScale, 10.0) * ScatteringFactor;
    
    float DarknessFactor = lerp(0.7, 0.5, DayNightFactor);
    SkyColor *= 1.0 - abs(1.0 - Dir.y) * DarknessFactor;
    
    return float4(SkyColor, 1.0);
}


float3 ScreenPosToWorldPos(float2 ScreenSpaceUV, float Depth, float4x4 ViewProjInv)
{
	float4 PosClipSpace;
    PosClipSpace.xy = ScreenSpaceUV * float2(2.0, -2.0) + float2(-1.0, 1.0);
    PosClipSpace.z = Depth;
    PosClipSpace.w = 1.0;
    float4 WorldPos = mul(PosClipSpace, ViewProjInv);
    return WorldPos.xyz / WorldPos.w;
}
