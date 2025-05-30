import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import urllib.parse
from Script.Design import constant, character_handle
from Script.Core import game_type, cache_control, flow_handle, main_frame, save_handle
from Script.UI.Flow import creator_character_flow

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """

# 全局变量，保存游戏状态
game_state = {}
available_actions = []
rewards = {}
now_ai_score = 0
add_score = 0

class GameRequestHandler(BaseHTTPRequestHandler):
    """
    处理HTTP请求的自定义请求处理器
    """

    def do_GET(self):
        """
        处理GET请求
        """
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path == '/restart':
            self.handle_restart()
        elif parsed_path.path == '/actions':
            self.handle_actions()
        elif parsed_path.path == '/state':
            self.handle_state()
        elif parsed_path.path == '/rewards':
            self.handle_rewards()
        elif parsed_path.path == '/current_panel':
            self.handle_current_panel()
        else:
            self.send_error(404, "接口未找到")

    def do_POST(self):
        """
        处理POST请求
        """
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path == '/step':
            self.handle_step()
        else:
            self.send_error(404, "接口未找到")

    def handle_restart(self):
        """
        处理/restart接口，重置游戏，返回初始状态
        """
        # 初始化游戏状态
        if cache.character_data[0].dead:
            while 1:
                if cache.now_panel_id != constant.Panel.TITLE:
                    main_frame.window.send_cmd("")
                    time.sleep(1)
                else:
                    break
        else:
            main_frame.window.send_cmd("94")
            time.sleep(1)
        while 1:
            if flow_handle.wait_switch:
                break
        main_frame.window.send_cmd("1")
        time.sleep(1)
        while 1:
            if flow_handle.wait_switch:
                break
        main_frame.window.send_cmd("0")
        time.sleep(1)
        while 1:
            if flow_handle.wait_switch:
                break
        main_frame.window.send_cmd("0")
        time.sleep(1)
        while 1:
            if flow_handle.wait_switch:
                break
        cache.back_save_panel = 1
        cache.wframe_mouse.w_frame_skip_wait_mouse = 1
        game_state = []
        for key in constant.handle_premise_data:
            game_state.append(constant.handle_premise_data[key](0))
        response = {'state': game_state}
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def handle_current_panel(self):
        """
        处理/current_panel接口，获取当前的面板
        """
        response = {'panel_id': cache.now_panel_id}
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        time.sleep(1)
        cache.back_save_panel = 1
        cache.wframe_mouse.w_frame_skip_wait_mouse = 1
        game_state = []
        for key in constant.handle_premise_data:
            game_state.append(constant.handle_premise_data[key](0))
        response = {'state': game_state}
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def handle_current_panel(self):
        """
        处理/current_panel接口，获取当前的面板
        """
        response = {'panel_id': cache.now_panel_id}
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def handle_step(self):
        """
        处理/step接口，执行动作，返回新的状态、奖励和是否结束
        """
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        print(post_data)
        data = json.loads(post_data.decode('utf-8'))
        action = data.get('action')
        if action not in set(flow_handle.cmd_map.keys()) and action != "":
            self.send_error(400, "无效的动作")
            return
        main_frame.window.input_event_func(action)
        time.sleep(1)
        cache.wframe_mouse.w_frame_skip_wait_mouse = 1
        while 1:
            if flow_handle.wait_switch:
                break
            time.sleep(0.1)
        self.update_score()
        # 检查游戏是否结束的逻辑
        done = False
        if cache.character_data[0].dead:
            done = True
        game_state = []
        for key in constant.handle_premise_data:
            game_state.append(constant.handle_premise_data[key](0))
        response = {
            'state': game_state,
            'reward': add_score,
            'done': done
        }
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def update_score(self):
        """ 更新积分 """
        global now_ai_score
        global add_score
        add_score = 0
        ai_score = 0
        if 0 not in cache.character_data:
            return
        player_data: game_type.Character = cache.character_data[0]
        for skill_id in player_data.knowledge:
            ai_score += player_data.knowledge[skill_id]
        for skill_id in player_data.language:
            ai_score += player_data.language[skill_id]
        for skill_id in player_data.sex_experience:
            ai_score += player_data.sex_experience[skill_id]
        for character_id in cache.character_data:
            if not character_id:
                continue
            character_data: game_type.Character = cache.character_data[character_id]
            if 0 in character_data.social_contact and character_data.social_contact[0] == 10:
                ai_score += 1
        for target_id in player_data.like_preference_data:
            ai_score += player_data.like_preference_data[target_id] * 1000
        for target_id in player_data.dislike_preference_data:
            ai_score -= player_data.dislike_preference_data[target_id] * 1000
        for friend_id in player_data.favorability:
            friend_data = cache.character_data[friend_id]
            if 0 in friend_data.favorability:
                ai_score += friend_data.favorability[0]
        ai_score += (cache.game_time - 1210629600) / 86400
        for achieve_id in cache_control.achieve.completed_data:
            if cache_control.achieve.completed_data[achieve_id]:
                ai_score += 10000
        if ai_score < 0:
            ai_score = 0
        if ai_score != now_ai_score:
            add_score = ai_score - now_ai_score
            now_ai_score = ai_score
        print("增加积分:", add_score, "当前积分:", now_ai_score)

    def handle_actions(self):
        """
        处理/actions接口，获取当前可用的命令列表
        """
        global available_actions
        cmd_map_set = {str(k) for k in flow_handle.cmd_map.keys()}
        if cache.now_panel_id == constant.Panel.IN_SCENE:
            cmd_map_set.discard(str(constant.Instruct.SAVE))
            cmd_map_set.discard(str(constant.Instruct.OBSERVE_ON))
            cmd_map_set.discard(str(constant.Instruct.OBSERVE_OFF))
        response = {'actions': list(cmd_map_set)}
        if len(cmd_map_set) == 0:
            main_frame.window.send_input()
            cache.wframe_mouse.mouse_leave_cmd = 1
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def handle_state(self):
        """
        处理/state接口，获取当前的世界状态
        """
        game_state = []
        for key in constant.handle_premise_data:
            game_state.append(constant.handle_premise_data[key](0))
        response = {'state': game_state}
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def handle_rewards(self):
        """
        处理/rewards接口，检查分数奖励
        """
        global rewards
        response = {'rewards': self.ai_score}
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

def run_server():
    server_address = ('', 5000)
    httpd = HTTPServer(server_address, GameRequestHandler)
    print('HTTP服务器已启动，正在监听端口5000...')
    httpd.serve_forever()
