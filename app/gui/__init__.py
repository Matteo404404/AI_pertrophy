def __init__(self, db_manager, tracking_system, user_manager):
        super().__init__()
        self.db = db_manager
        self.tracking_system = tracking_system
        self.user_manager = user_manager
        
        # Initialize AI
        if ML_AVAILABLE:
            # We point to the model file in the sibling directory
            model_path = "ml_engine/models/strength_predictor.pt"
            self.ai_predictor = HybridPredictor(self.db, model_path)
        else:
            self.ai_predictor = None
            
        self.init_ui()