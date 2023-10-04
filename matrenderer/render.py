import torch
import torch.nn as nn

class Light:
    """Point light settings
    """
    def __init__(self, position: tuple[float, float, float], color: tuple[float, float, float], power: float):
        self.position = torch.tensor(position)
        self.color = torch.tensor(color) * power

    def to(self, device: torch.device):
        self.position = self.position.to(device)
        self.color = self.color.to(device)
        return self

class Renderer(nn.Module):
    """Differentiable physics-based material renderer using svBRDF maps.
    """

    def __init__(
            self, 
            size: float = 10.0, 
            camera: tuple[float] = [0.0, 0.0, 10.0],
            lights: list[Light] = [Light((0.0, 0.0, 10.0), (23.47, 21.31, 20.79), 10.0)],
            ambient: float = 0.3,
            F0: float = 0.04,
            gamma: float = 2.2,
            eps: float = 1e-8,
        ):
        """Initialize scene setup.

        Args:
            size (float, optional): Real-world size of the texture plane. Defaults to 30.0.
            camera (List[float], optional): Position of the camera relative to the texture center.
                The texture always resides on the X-Y plane in center alignment.
                Defaults to [0.0, 0.0, 25.0].
            lights (List[Light], optional): List of point lights in the scene.
            ambient (float, optional): Strength of ambient light.
            F0 (float, optional): Normalized ambient light intensity. Defaults to 0.04.
            gamma (float, optional): Gamma value used for gamma correction. Defaults to 2.2.
        """
        super(Renderer, self).__init__()
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.size = size
        self.camera = torch.tensor(camera).to(self.device)
        self.F0 = F0
        self.gamma = gamma
        self.ambient = torch.tensor([ambient, ambient, ambient]).to(self.device)
        self.eps = eps
        self.lights = [l.to(self.device) for l in lights]

    def check_maps(self, maps: {str: torch.tensor}) -> {str: torch.tensor}:
        assert maps['basecolor'] != None
        shape = maps['basecolor'].shape
        if maps['normal'] == None:
            normal = torch.zeros_like(maps['basecolor'])
            normal[:,:,2] = 1.0
            maps['normal'] = normal
        if maps['roughness'] == None:
            maps['roughness'] = torch.zeros((shape[0], 1, *shape[2:]), device=maps['basecolor'].device)
        if maps['metallic'] == None:
            maps['metallic'] = torch.zeros((shape[0], 1, *shape[2:]), device=maps['basecolor'].device)
        if maps['ao'] == None:
            maps['ao'] = torch.ones((shape[0], 1, *shape[2:]), device=maps['basecolor'].device)
        if maps['height'] == None:
            maps['height'] = torch.zeros((shape[0], 1, *shape[2:]), device=maps['basecolor'].device)
        return maps

    def gamma_correction(self, input) -> torch.Tensor:
        return input ** (1.0 / self.gamma)
    
    def get_positions(self, albedo):
        img_size: int = albedo.shape[2]
        x_coords = torch.linspace(0.5 / img_size - 0.5, 0.5 - 0.5 / img_size, img_size,
                               device=self.device)
        x_coords = x_coords * self.size

        x, y = torch.meshgrid(x_coords, x_coords, indexing='xy')
        pos = torch.stack((x, -y, torch.zeros_like(x)))
        return pos

    def render(self, maps: {str: torch.tensor}) -> torch.Tensor:
        """Render image based on scene and given svBRDF maps.

        Args:
            maps (dict, {std: tensor}): Material svBRDF maps dictionary for rendering.
        Returns:
            Tensor: Rendered image.

        """
        maps = self.check_maps(maps)

        albedo = maps['basecolor'].to(self.device) ** self.gamma
        normal = maps['normal'].to(self.device)
        roughness = maps['roughness'].to(self.device)
        metallic = maps['metallic'].to(self.device)
        height = maps['height'].to(self.device)
        ao = maps['ao'].to(self.device)
        batch_size = albedo.size(0)

        """
        # Discard the alpha channel of basecolor and normal, map the basecolor to gamma space, and
        # scale the normal image to [-1, 1]
        albedo = albedo.narrow(1, 0, 3) ** 2.2
        normal = ((normal.narrow(1, 0, 3) - 0.5) * 2.0)
        """
        camera      = self.camera.view(3, 1, 1)
        pos         = self.get_positions(albedo)
        # intermediate lighting buffers
        Lo          = torch.zeros_like(albedo)
        t_radiance  = torch.zeros_like(albedo)
        diffuse     = torch.zeros_like(albedo)
        specular    = torch.zeros_like(albedo)

        F0 = self.F0 * torch.ones_like(metallic)
        F0 = torch.lerp(F0, albedo, metallic)
        # Nx3xWxH
        normal = ((normal.narrow(1, 0, 3) - 0.5) * 2.0)
        N = normal/torch.norm(normal, dim=1, keepdim=True)                      # normal
        # 3xWxH for all data in the batch
        V = (camera - pos) / torch.norm(camera - pos, dim=0, keepdim=True)      # view direcion

        for light in self.lights:
            light_pos   = light.position.view(3, 1, 1)
            light_color = light.color.view(3, 1, 1)
            distance    = torch.norm(light_pos - pos, dim=0)

            L = (light_pos - pos) / distance                                    # light vector
            H = (V + L) / torch.norm(V + L, dim=0, keepdim=True)                # half vector
            
            attenuation = 1.0 / (distance ** 2)
            radiance    = light_color * attenuation

            # expanding vectors to cover entrie batch
            V = V.unsqueeze(0).repeat(batch_size, 1, 1, 1)                      # 3xWxH -> Bx3xWxH
            H = H.unsqueeze(0).repeat(batch_size, 1, 1, 1)                      # 3xWxH -> Bx3xWxH
            L = L.unsqueeze(0).repeat(batch_size, 1, 1, 1)                      # 3xWxH -> Bx3xWxH
            radiance = radiance.unsqueeze(0).repeat(batch_size, 1, 1, 1)        # 3xWxH -> Bx3xWxH
            t_radiance += radiance

            NDF = DistributionGGX(N, H, roughness)                              # normal distribution function
            G   = GeometrySmith(N, V, L, roughness)                             # geometry
            F   = fresnelSchlick(torch.clamp_min(vector_dot(H, V), 0.0), F0)    # fresnel

            # Cook-Torrance specular BRDF
            numerator   = NDF * G * F
            denominator = 4.0 * torch.clamp_min(vector_dot(N, V), 0.0) * torch.clamp_min(vector_dot(N, L), 0.0) + self.eps
            specular   += numerator / denominator
            kS = F
            # Diffuse kD = 1.0 - kS
            kD = 1.0 - kS
            kD *= 1.0 - metallic; 
            diffuse += kD * albedo / torch.pi
        
            NdotL = torch.clamp_min(vector_dot(N, L), 0.0)
            Lo += (diffuse + specular) * radiance * NdotL

        ambient     = self.ambient.view(3, 1, 1) * albedo * ao
        color       = torch.clamp(ambient + Lo, min=self.eps, max=1.0)

        color = self.gamma_correction(color)

        return color, ambient, t_radiance, diffuse, specular

# A, B: (Bx3x...) only for unit vectors
def vector_dot(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.clamp((A * B).sum(1, keepdim=True), min=0.0, max=1.0)

# N, H: （3xWxH), roughness: (Bx1xWxH)
def DistributionGGX(N: torch.Tensor, H: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:

    a   = roughness*roughness
    a2  = a*a
    NdotH = vector_dot(N, H)
    NdotH2 = NdotH*NdotH
	
    num   = a2
    denom = (NdotH2 * (a2 - 1.0) + 1.0)
    denom = torch.pi * denom * denom

    return num / denom

# NdotV, roughness: (Bx1xWxH)
def GeometrySchlickGGX(NdotV: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    r = (roughness + 1.0)
    k = (r*r) / 8.0

    num   = NdotV
    denom = NdotV * (1.0 - k) + k
	
    return num / denom

# N, V, L: （Bx3xWxH), roughness: (Bx1xWxH)
def GeometrySmith(N: torch.Tensor, V: torch.Tensor, L: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    NdotV = vector_dot(N, V)
    NdotL = vector_dot(N, L)
    ggx2  = GeometrySchlickGGX(NdotV, roughness)
    ggx1  = GeometrySchlickGGX(NdotL, roughness)
	
    return ggx1 * ggx2

# cosTheta, F0 (Bx1xWxH)
def fresnelSchlick(cosTheta: torch.Tensor, F0: torch.Tensor) -> torch.Tensor:
    return F0 + (1.0 - F0) * torch.pow(torch.clamp(1.0 - cosTheta, min=0.0, max=1.0), 5.0);
