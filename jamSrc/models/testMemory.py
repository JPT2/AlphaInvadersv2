import unittest
import jamSrc.agentMemory as mem


class TestAgentMemoryBestEpisode(unittest.TestCase):
    def should_create_dataset_from_data(self):
        pass

    def test_should_add_episode_to_empty_best_episodes(self):
        memory = mem.Memory()
        memory.update_best_episodes([], [], [], total_rewards=1)
        self.assertEqual(memory.best_episodes[0]['total_rewards'], 1)

    def test_should_have_sorted_list_after_adding_5th_episode(self):
        memory = mem.Memory()
        memory.update_best_episodes([], [], [], total_rewards=4)
        memory.update_best_episodes([], [], [], total_rewards=1)
        memory.update_best_episodes([], [], [], total_rewards=5)
        memory.update_best_episodes([], [], [], total_rewards=2)
        memory.update_best_episodes([], [], [], total_rewards=3)

        self.assertEqual(memory.best_episodes[0]['total_rewards'], 1)
        self.assertEqual(memory.best_episodes[1]['total_rewards'], 2)
        self.assertEqual(memory.best_episodes[2]['total_rewards'], 3)
        self.assertEqual(memory.best_episodes[3]['total_rewards'], 4)
        self.assertEqual(memory.best_episodes[4]['total_rewards'], 5)

    def test_should_not_add_episode_with_lower_total_rewards_to_best_episode(self):
        memory = mem.Memory()
        memory.update_best_episodes([], [], [], total_rewards=1)
        memory.update_best_episodes([], [], [], total_rewards=2)
        memory.update_best_episodes([], [], [], total_rewards=3)
        memory.update_best_episodes([], [], [], total_rewards=4)
        memory.update_best_episodes([], [], [], total_rewards=5)

        negative_reward = -100
        memory.update_best_episodes([], [], [], total_rewards=negative_reward)
        self.assertEqual(memory.best_episodes[0]['total_rewards'], 1)

    def test_should_add_episode_with_better_total_rewards_to_best_episode(self):
        memory = mem.Memory()
        memory.update_best_episodes([], [], [], total_rewards=1)
        memory.update_best_episodes([], [], [], total_rewards=2)
        memory.update_best_episodes([], [], [], total_rewards=3)
        memory.update_best_episodes([], [], [], total_rewards=4)
        memory.update_best_episodes([], [], [], total_rewards=5)

        largest_reward = 100
        memory.update_best_episodes([], [], [], total_rewards=largest_reward)
        self.assertEqual(memory.best_episodes[4]['total_rewards'], largest_reward)

    def test_should_remain_sorted_on_new_addition(self):
        memory = mem.Memory()
        memory.update_best_episodes([], [], [], total_rewards=1)
        memory.update_best_episodes([], [], [], total_rewards=2)
        memory.update_best_episodes([], [], [], total_rewards=3)
        memory.update_best_episodes([], [], [], total_rewards=4)
        memory.update_best_episodes([], [], [], total_rewards=5)

        middle_reward = 3.5
        memory.update_best_episodes([], [], [], total_rewards=middle_reward)

        expected_rewards = [2, 3, middle_reward, 4, 5]
        for i in range(len(expected_rewards)):
            self.assertEqual(memory.best_episodes[i]['total_rewards'], expected_rewards[i])


if __name__ == '__main__':
    unittest.main()
