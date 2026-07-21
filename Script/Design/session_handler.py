from typing import Dict, List, Any
from Script.Core import cache_control, game_type
from Script.Design import constant, map_handle

cache: game_type.Cache = cache_control.cache

class BaseSessionHandler:
    """ 基础会话处理器接口 """
    
    def __init__(self, session_uid: str):
        self.session_uid = session_uid
        
    @property
    def session(self):
        return cache.interaction_sessions.get(self.session_uid)
        
    def on_start(self):
        """ 会话成功建立时的初始化处理 """
        pass
        
    def on_update(self) -> bool:
        """ 会话每 Tick/每分钟 的演进逻辑, 返回 True 表示结束 """
        return False
        
    def on_member_join(self, character_id: int):
        """ 新成员加入时的处理 """
        pass
        
    def on_member_leave(self, character_id: int):
        """ 成员离开时的处理 """
        pass
        
    def on_finish(self):
        """ 会话结束时的结算处理 """
        pass

class ChatSessionHandler(BaseSessionHandler):
    """ 闲聊会话处理器 """
    
    def on_start(self):
        session = self.session
        if not session:
            return
        session.data.setdefault('stage', 0)
        session.data.setdefault('atmosphere', 50)
        
    def on_update(self):
        import random
        session = self.session
        if not session:
            return True
            
        elapsed_time = cache.game_time - session.start_time
        # 每5分钟推进一个阶段
        if elapsed_time > 5 * 60:
            stage = session.data.get('stage', 0)
            if stage == 0:
                session.data['atmosphere'] += random.randint(-10, 20)
                session.data['stage'] = 1
                session.start_time = cache.game_time # 重置计时
            elif stage == 1:
                if session.data['atmosphere'] > 60:
                    self._settle_chat_effect()
                session.data['atmosphere'] += random.randint(-10, 20)
                session.data['stage'] = 2
                session.start_time = cache.game_time
            elif stage >= 2:
                return True # 返回 True 表示会话应结束
        return False
        
    def _settle_chat_effect(self):
        """ 闲聊阶段结算效果 """
        session = self.session
        if not session:
            return
        for member_id in session.members:
            member_data = cache.character_data.get(member_id)
            if member_data:
                member_data.status.setdefault(25, 0)
                member_data.status[25] = max(0, member_data.status[25] - 1) # Reduce fatigue
                
    def on_finish(self):
        # 闲聊结束后的清理
        pass

class ClassroomSessionHandler(BaseSessionHandler):
    """ 课堂教学会话 """
    def on_start(self):
        session = self.session
        if not session:
            return
        session.data.setdefault('atmosphere', 50)
        
    def on_update(self) -> bool:
        session = self.session
        if not session:
            return True
            
        elapsed_time = cache.game_time - session.start_time
        if elapsed_time > 60: # Every game minute
            # Handle tick logic
            initiator_data = cache.character_data.get(session.initiator)
            if initiator_data:
                initiator_data.status.setdefault(25, 0)
                initiator_data.status[25] += 0.05 # Exhaustion
            
            # Students learn
            course_id = session.data.get('course')
            for member_id in session.members:
                if member_id == session.initiator:
                    continue
                member_data = cache.character_data.get(member_id)
                if member_data:
                    # e.g., improve course knowledge
                    pass
            session.start_time = cache.game_time
            
            # Check if teacher stopped teaching
            if initiator_data and initiator_data.state != constant.CharacterStatus.STATUS_SOCIAL_INTERACTING:
                return True
                
        return False

class DiningSessionHandler(BaseSessionHandler):
    """ 多人聚餐会话 """
    def on_start(self):
        session = self.session
        if not session:
            return
        session.data.setdefault('atmosphere', 60)
        
    def on_update(self) -> bool:
        session = self.session
        if not session:
            return True
            
        elapsed_time = cache.game_time - session.start_time
        if elapsed_time > 60: # Every game minute
            import random
            
            # Settle hunger and thirst for members
            for member_id in session.members:
                member_data = cache.character_data.get(member_id)
                if member_data:
                    member_data.status.setdefault(27, 0) # Hunger
                    member_data.status.setdefault(28, 0) # Thirst
                    member_data.status[27] = max(0, member_data.status[27] - 2)
                    member_data.status[28] = max(0, member_data.status[28] - 2)
                    
                    # Random topic interest boost
                    if random.random() < 0.2:
                        session.data['atmosphere'] = min(100, session.data['atmosphere'] + 5)
                        
            session.start_time = cache.game_time
            
            # Check initiator state
            initiator_data = cache.character_data.get(session.initiator)
            if initiator_data and initiator_data.state != constant.CharacterStatus.STATUS_SOCIAL_INTERACTING:
                return True
                
        return False

class GameSessionHandler(BaseSessionHandler):
    """ 博弈与娱乐会话 (如下棋、打牌) """
    def on_start(self):
        session = self.session
        if not session:
            return
        session.data.setdefault('atmosphere', 70)
        session.data.setdefault('scores', {m: 0 for m in session.members})
        session.data.setdefault('turn_owner', session.initiator)
        
    def on_update(self) -> bool:
        session = self.session
        if not session:
            return True
            
        elapsed_time = cache.game_time - session.start_time
        if elapsed_time > 60: # Every game minute
            import random
            
            # Simple game logic: random score increment for turn owner
            turn_owner = session.data.get('turn_owner')
            scores = session.data.setdefault('scores', {})
            
            if turn_owner in scores:
                scores[turn_owner] += random.randint(1, 5)
                
            # Switch turn
            members_list = session.members
            if members_list:
                current_idx = members_list.index(turn_owner) if turn_owner in members_list else 0
                next_idx = (current_idx + 1) % len(members_list)
                session.data['turn_owner'] = members_list[next_idx]
                
            # Settle entertainment value
            for member_id in session.members:
                member_data = cache.character_data.get(member_id)
                if member_data:
                    member_data.status.setdefault(25, 0)
                    member_data.status[25] = max(0, member_data.status[25] - 1) # Reduce fatigue
                    
            session.start_time = cache.game_time
            
            # Win condition
            for member_id, score in scores.items():
                if score >= 50:
                    return True # Game over
                    
            # Check if members left
            initiator_data = cache.character_data.get(session.initiator)
            if initiator_data and initiator_data.state != constant.CharacterStatus.STATUS_SOCIAL_INTERACTING:
                return True
                
        return False

class ArgumentSessionHandler(BaseSessionHandler):
    """ 争吵会话 """
    def on_start(self):
        session = self.session
        if not session:
            return
        session.data.setdefault('atmosphere', 0) # Very low atmosphere
        
    def on_update(self) -> bool:
        session = self.session
        if not session:
            return True
            
        elapsed_time = cache.game_time - session.start_time
        if elapsed_time > 60: # Every game minute
            
            # Radiate stress to others nearby
            initiator_data = cache.character_data.get(session.initiator)
            if initiator_data:
                scene_path_str = map_handle.get_map_system_path_str_for_list(initiator_data.position)
                scene_data = cache.scene_data.get(scene_path_str)
                if scene_data:
                    for char_id in scene_data.character_list:
                        if char_id not in session.members:
                            bystander = cache.character_data.get(char_id)
                            if bystander:
                                bystander.status.setdefault(25, 0)
                                bystander.status[25] += 0.1 # Stress / exhaustion increase for bystanders
                                
            # Settle exhaustion for members
            for member_id in session.members:
                member_data = cache.character_data.get(member_id)
                if member_data:
                    member_data.status.setdefault(25, 0)
                    member_data.status[25] += 0.5
                    
            session.start_time = cache.game_time
            
            # Check initiator state
            if initiator_data and initiator_data.state != constant.CharacterStatus.STATUS_SOCIAL_INTERACTING:
                return True
                
        return False

# 会话处理器工厂/注册表
SESSION_HANDLERS = {
    constant.Behavior.CHAT: ChatSessionHandler,
    constant.Behavior.TEACHING: ClassroomSessionHandler,
    constant.Behavior.EAT: DiningSessionHandler,
    constant.Behavior.PLAY_COMPUTER: GameSessionHandler,
    constant.Behavior.ABUSE: ArgumentSessionHandler,
    # 后续可注册更多的 Handler
}

def get_session_handler(session_uid: str):
    session = cache.interaction_sessions.get(session_uid)
    if not session:
        return None
    handler_class = SESSION_HANDLERS.get(session.type, BaseSessionHandler)
    return handler_class(session_uid)
