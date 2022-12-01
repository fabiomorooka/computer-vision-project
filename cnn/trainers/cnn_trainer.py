# encoding: utf-8
import torch
from tqdm import tqdm


class CNNTrainModel():
	def __init__(self, model, criterion, optimizer, train_dataloader, valid_dataloader, run, file_name):
		self.model = model.double().train()
		self.criterion = criterion
		self.optimizer = optimizer
		self.train_dataloader = train_dataloader
		self.valid_dataloader = valid_dataloader
		self.run = run
		self.file_name = file_name

	def train(self, epochs, device):
		best_valid_loss = 10e9
		best_epoch = 0
		train_loss = 0.0

		for epoch in range(epochs+1):
			accumulated_loss = 0
			accumulated_accuracy = 0
			for x_train, y_train in tqdm(self.train_dataloader):
				batch_loss = 0.0
				x_train = x_train.double().to(device)
				y_train = y_train.long().to(device)

				# Clear all accumulated gradients
				self.optimizer.zero_grad()

				# Forward pass
				pred_y = self.model(x_train)

				# Compute loss
				batch_loss = self.criterion(pred_y, y_train)

				# Backpropagate the loss
				batch_loss.backward()
				
				# Adjust parameters according to the computed gradients
				self.optimizer.step()
				accumulated_loss += batch_loss.item()
				_, prediction = torch.max(pred_y, 1)
				self.run['train/batch_loss'].log(batch_loss.item())

				accumulated_accuracy += torch.sum(prediction == y_train)
			
			train_loss = accumulated_loss / len(self.train_dataloader.dataset)
			train_acc = accumulated_accuracy / len(self.train_dataloader.dataset)
			self.run['train/loss'].log(train_loss)
			
			# validation loop
			accumulated_loss = 0
			accumulated_accuracy = 0
			self.model.eval()
			for x_valid, y_valid in tqdm(self.valid_dataloader):
				batch_loss = 0.0
				x_valid = x_valid.double().to(device)
				y_valid = y_valid.long().to(device)

				with torch.no_grad():
					# Forward pass
					pred_y = self.model(x_valid)
					_, prediction = torch.max(pred_y, 1)

					# Compute loss
					batch_loss = self.criterion(pred_y, y_valid)
					accumulated_loss += batch_loss.item()

					# Compute accuracy
					accumulated_accuracy += torch.sum(prediction == y_valid)

			valid_loss = accumulated_loss / len(self.valid_dataloader.dataset)
			valid_acc = accumulated_accuracy / len(self.valid_dataloader.dataset)
			self.run['valid/loss'].log(valid_loss)
			self.run['valid/acuracy'].log(valid_acc)
					
			print(f'Epoch: {epoch:d}/{epochs:d} Train Loss: {train_loss:.6f} | Train accuracy = {100*train_acc:.6f}% | Valid Loss: {valid_loss:.6f} | Val accuracy = {100*valid_acc:.6f}%')

			# Salvando o melhor modelo de acordo com a loss de validação
			if valid_loss < best_valid_loss:
				torch.save(self.model.state_dict(), self.file_name+'.pt')
				best_valid_loss = valid_loss
				best_accuracy = valid_acc
				best_epoch = epoch+1
				print(f'best model')

		print(f'########### FINAL RESULTS ####################')
		print(f'Final loss: {best_valid_loss}')
		print(f'Final accuracy: {100*best_accuracy:.6f}%')
		print(f'at epoch:', best_epoch)
		print(f'##############################################')

	def evaluate(self, test_dataloader, device):
		self.model.load_state_dict(torch.load(self.file_name+'.pt'))
		accumulated_loss = 0
		accumulated_accuracy = 0
		self.model.eval()
		for x_test, y_test in tqdm(test_dataloader):
			x_test = x_test.double().to(device)
			y_test = y_test.long().to(device)

			with torch.no_grad():
				# Forward pass
				pred_y = self.model(x_test)
				_, prediction = torch.max(pred_y, 1)

				# Compute loss
				batch_loss = self.criterion(pred_y, y_test)
				accumulated_loss += batch_loss.item()

				# Compute accuracy
				accumulated_accuracy += torch.sum(prediction == y_test)

		test_loss = accumulated_loss / len(test_dataloader.dataset)
		test_acc = accumulated_accuracy / len(test_dataloader.dataset)
		print(f'The final accuracy of the model using the (reduced) test dataset is: {100*test_acc:.6f}%')
		return test_loss, test_acc
