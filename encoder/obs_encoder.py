import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class ObsEncoder(nn.Module):
    def __init__(self, args):
        super(ObsEncoder, self).__init__()
        self.state_dim = 0
        self.args = args
        if args.obs_goal:
            self.goal_encoder = GoalEncoder(args)
            self.state_dim += args.feature_dim
        if args.obs_waypoints:
            self.waypoints_encoder = WayPointsEncoder(args)
            self.state_dim += args.feature_dim
        if args.obs_map:
            self.map_encoder = MapEncoder(args)
            self.state_dim += args.feature_dim
        if args.obs_previous_action:
            self.previous_action_encoder = ActionEncoder(args)
            self.state_dim += args.feature_dim
        self.fc = nn.Linear(self.state_dim, args.feature_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_goal=None, x_waypoints=None, x_lidar=None, x_action=None):
        state = []
        if x_goal is not None:
            state.append(self.goal_encoder(x_goal))
        if x_waypoints is not None:
            state.append(self.waypoints_encoder(x_waypoints))
        if x_lidar is not None:
            state.append(self.map_encoder(x_lidar))
        if x_action is not None:
            state.append(self.previous_action_encoder(x_action))
        x =  torch.cat(state, -1)
        
        x = self.fc(x)
        x = self.relu(x)
        return x

class GoalEncoder(nn.Module):
    def __init__(self, args):
        super(GoalEncoder, self).__init__()
        if args.env == "iGibson":
            input_size = 4
        else:
            input_size = 2
        self.fc = nn.Linear(input_size, args.feature_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

class WayPointsEncoder(nn.Module):
    def __init__(self, args):
        super(WayPointsEncoder, self).__init__()        
        self.fc = nn.Linear(2*args.num_wps_input, args.feature_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

class ActionEncoder(nn.Module):
    def __init__(self, args):
        super(ActionEncoder, self).__init__()
        input_size = 2
        self.fc = nn.Linear(input_size, args.feature_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

class MapEncoder(nn.Module):
    def __init__(self, args):
        super(MapEncoder, self).__init__()
        self.args = args
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 256)
        self.resnet.eval()
        self.relu = nn.ReLU(inplace=True)

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),                     # Convert to PIL image
            transforms.Resize((224, 224)),                # Resize to 224x224 pixels
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB
            transforms.ToTensor(),                        # Convert to tensor
            transforms.Normalize((0.5,), (0.5,))          # Normalize the image
        ])

    def forward(self, x):
        x = self.preprocess(x).to(self.args.device).unsqueeze(0)
        with torch.no_grad():
            x = self.resnet(x)
        x = self.relu(x)
        return x

